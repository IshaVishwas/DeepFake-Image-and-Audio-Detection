import os
import time
import torch
import torch.nn as nn
import torchvision.models as models
import cv2
import numpy as np
import librosa

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from pydub import AudioSegment

app = Flask(__name__)

# ---------------- CONFIG ----------------

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

IMG_SIZE = 128
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BEST_THRESH_IMAGE = 0.45
BEST_THRESH_AUDIO = 0.95


# ---------------------------------------------------------
# IMAGE MODEL
# ---------------------------------------------------------

class SmallCNN(nn.Module):
    def __init__(self, in_ch=1, out_dim=64, variant="pool2"):
        super().__init__()

        if variant == "pool2":
            conv = nn.Sequential(
                nn.Conv2d(in_ch, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.MaxPool2d(2),
            )
        else:
            conv = nn.Sequential(
                nn.Conv2d(in_ch, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.MaxPool2d(2),
            )

        self.net = nn.Sequential(
            conv,
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, out_dim),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.net(x)


class ResNet18Backbone(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        base = models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.proj = nn.Sequential(
            nn.Linear(512, out_dim),
            nn.ReLU(True),
            nn.Dropout(0.4)
        )

    def forward(self, x):
        return self.proj(self.backbone(x).flatten(1))


class AttentionGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(True),
            nn.Linear(dim // 4, dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.gate(x)


class ImageDeepfakeDetector(nn.Module):
    FEAT_DIM = 448

    def __init__(self):
        super().__init__()

        self.diffusion_branch = SmallCNN(1, 64)
        self.edge_branch = SmallCNN(1, 64, "convconvpool")
        self.semantic_branch = ResNet18Backbone(256)
        self.acquisition_branch = SmallCNN(1, 64)
        self.attention = AttentionGate(self.FEAT_DIM)

        self.classifier = nn.Sequential(
            nn.Linear(self.FEAT_DIM, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(0.6),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, d, e, s, a):
        f1 = self.diffusion_branch(d)
        f2 = self.edge_branch(e)
        f3 = self.semantic_branch(s)
        f4 = self.acquisition_branch(a)

        feats = torch.cat([f1, f2, f3, f4], dim=1)
        feats = self.attention(feats)

        return self.classifier(feats), feats


# ---------------------------------------------------------
# AUDIO MODEL
# ---------------------------------------------------------

class AudioModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.lstm = nn.LSTM(64 * 16, 64, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(128, 1)

        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.cnn(x)
        B, C, H, W = x.shape
        x = x.view(B, W, C * H)

        x, _ = self.lstm(x)

        a = torch.softmax(self.attn(x).squeeze(-1), dim=1)
        x = torch.sum(x * a.unsqueeze(-1), dim=1)

        return self.fc(x)


# ---------------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------------

image_model = ImageDeepfakeDetector().to(DEVICE)
audio_model = AudioModel().to(DEVICE)

IMAGE_PATH = "models/image_model.pt"
AUDIO_PATH = "models/audio_model.pth"

if os.path.exists(IMAGE_PATH):
    ckpt = torch.load(IMAGE_PATH, map_location=DEVICE)
    image_model.load_state_dict(ckpt["model_state"])
    image_model.eval()
    print("✅ Image model loaded")

if os.path.exists(AUDIO_PATH):
    ckpt = torch.load(AUDIO_PATH, map_location=DEVICE)
    audio_model.load_state_dict(ckpt["model_state"])
    audio_model.eval()
    print("✅ Audio model loaded")


# ---------------------------------------------------------
# AUDIO PREPROCESS
# ---------------------------------------------------------

def convert_to_wav(input_path):
    ext = input_path.split('.')[-1].lower()

    if ext == "wav":
        return input_path

    audio = AudioSegment.from_file(input_path)
    output_path = input_path.rsplit('.', 1)[0] + "_converted.wav"

    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(output_path, format="wav")

    return output_path


def preprocess_audio(path):
    wav_path = convert_to_wav(path)

    y, sr = librosa.load(wav_path, sr=16000)

    L = sr * 3
    if len(y) < L:
        y = np.pad(y, (0, L - len(y)))
    else:
        y = y[:L]

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel)

    pitch = librosa.yin(y, fmin=50, fmax=300)
    energy = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]

    T = mel_db.shape[1]

    def fix(x):
        return np.pad(x, (0, T - len(x)))[:T]

    features = np.vstack([mel_db, fix(pitch), fix(energy), fix(zcr)])

    return torch.tensor(features, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE), wav_path


# ------------------------------------------------------------------
#  img preprocess
# -----------------------------------------------------------------

def normalize_01(x):
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-8)


def preprocess_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    u8 = (gray * 255).astype(np.uint8)

    fft = normalize_01(np.abs(np.fft.fftshift(np.fft.fft2(gray))))[None]
    edge = (cv2.Canny(u8, 100, 200).astype(np.float32) / 255.0)[None]

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = ((rgb - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)

    lap = normalize_01(cv2.Laplacian(u8, cv2.CV_64F).astype(np.float32))[None]

    # return (
    #     torch.tensor(fft).unsqueeze(0).to(DEVICE),
    #     torch.tensor(edge).unsqueeze(0).to(DEVICE),
    #     torch.tensor(rgb).unsqueeze(0).to(DEVICE),
    #     torch.tensor(lap).unsqueeze(0).to(DEVICE),
    # )
    return (
        torch.tensor(fft, dtype=torch.float32).unsqueeze(0).to(DEVICE),
        torch.tensor(edge, dtype=torch.float32).unsqueeze(0).to(DEVICE),
        torch.tensor(rgb, dtype=torch.float32).unsqueeze(0).to(DEVICE),
        torch.tensor(lap, dtype=torch.float32).unsqueeze(0).to(DEVICE),
    )


# ---------------------------------------------------------
# ROUTES
# ---------------------------------------------------------

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files['file']
    mode = request.form.get('mode')

    path = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
    file.save(path)

    try:
        # ---------------- IMAGE ----------------
        if mode == 'image':
            d, e, s, a = preprocess_image(path)

            with torch.no_grad():
                logits, _ = image_model(d, e, s, a)   # 🔥 FIX
                score = torch.sigmoid(logits).item()

            status = "Deepfake" if score > BEST_THRESH_IMAGE else "Real"
            confidence = round((score if status == "Deepfake" else 1 - score) * 100, 2)

            return jsonify({"status": status, "confidence": f"{confidence}%"})

        # ---------------- AUDIO (UNCHANGED) ----------------
        elif mode == 'audio':
            x, wav_path = preprocess_audio(path)

            with torch.no_grad():
                score = torch.sigmoid(audio_model(x)).item()

            status = "Deepfake" if score > BEST_THRESH_AUDIO else "Real"
            confidence = round((score if status == "Deepfake" else 1 - score) * 100, 2)

            return jsonify({"status": status, "confidence": f"{confidence}%"})

        else:
            return jsonify({"error": "Invalid mode"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
