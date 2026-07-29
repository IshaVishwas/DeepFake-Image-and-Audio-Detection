const toggle = document.getElementById('typeToggle');
const fileInput = document.getElementById('fileInput');
const dropzone = document.getElementById('dropzone');
const promptText = document.getElementById('promptText');
const iconBox = document.getElementById('icon-box');
const submitBtn = document.getElementById('submitBtn');
const loader = document.getElementById('loader');
const resultCard = document.getElementById('resultCard');

let currentMode = 'audio'; // default

// ---------------- MODE TOGGLE ----------------
toggle.addEventListener('change', () => {

```
resultCard.classList.add('hidden');
fileInput.value = "";

if(toggle.checked) {
    currentMode = 'image';
    promptText.innerText = "Drop your image here";
    iconBox.innerText = "🖼️";
    fileInput.accept = "image/*";
} else {
    currentMode = 'audio';
    promptText.innerText = "Drop your audio file here";
    iconBox.innerText = "🎵";
    fileInput.accept = ".wav,.mp3,.flac"; // 🔥 FIX
}
```

});

// ---------------- CLICK UPLOAD ----------------
dropzone.addEventListener('click', () => fileInput.click());

// ---------------- FILE SELECT ----------------
fileInput.addEventListener('change', () => {
if(fileInput.files.length > 0) {
promptText.innerText = fileInput.files[0].name;
promptText.classList.add('text-blue-400');
}
});

// ---------------- SUBMIT ----------------
submitBtn.addEventListener('click', async () => {

```
if(!fileInput.files[0]) {
    alert("Please select a file first!");
    return;
}

const formData = new FormData();
formData.append('file', fileInput.files[0]);
formData.append('mode', currentMode); // 🔥 FIXED

loader.classList.remove('hidden');
resultCard.classList.add('hidden');
submitBtn.disabled = true;

try {
    const response = await fetch('/analyze', {
        method: 'POST',
        body: formData
    });

    let data;
    try {
        data = await response.json();
    } catch {
        throw new Error("Invalid server response");
    }

    console.log("API RESPONSE:", data);

    loader.classList.add('hidden');
    resultCard.classList.remove('hidden');

    // 🔥 HANDLE ERRORS PROPERLY
    if (!response.ok || data.error) {
        document.getElementById('resStatus').innerText = "Error";
        document.getElementById('resConf').innerText = data.error || "Server error";
        document.getElementById('resStatus').className = "text-3xl font-black text-yellow-400";
        return;
    }

    document.getElementById('resStatus').innerText = data.status;
    document.getElementById('resConf').innerText = data.confidence;

    if(data.status === "Deepfake") {
        document.getElementById('resStatus').className = "text-3xl font-black text-red-500";
        resultCard.style.borderColor = "#ef4444";
    } else {
        document.getElementById('resStatus').className = "text-3xl font-black text-green-500";
        resultCard.style.borderColor = "#22c55e";
    }

} catch (error) {
    console.error(error);
    alert("Server error: " + error.message);
    loader.classList.add('hidden');
} finally {
    submitBtn.disabled = false;
}
```

});

// ---------------- DRAG DROP ----------------
dropzone.addEventListener('dragover', (e) => {
e.preventDefault();
dropzone.classList.add('border-blue-400');
});

dropzone.addEventListener('dragleave', () => {
dropzone.classList.remove('border-blue-400');
});

dropzone.addEventListener('drop', (e) => {
e.preventDefault();
dropzone.classList.remove('border-blue-400');
fileInput.files = e.dataTransfer.files;

```
if(fileInput.files[0]) {
    promptText.innerText = fileInput.files[0].name;
}
```

});
