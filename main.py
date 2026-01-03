<!DOCTYPE html>
<html>
<head>
    <title>Privalens Secure ID</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #1a1a2e; color: white; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; }
        .box { background: #16213e; padding: 40px; border-radius: 15px; text-align: center; box-shadow: 0 10px 30px rgba(0,0,0,0.5); width: 350px; }
        h2 { margin-bottom: 20px; color: #0f3460; text-shadow: 0 0 5px #e94560; color: #fff; }
        video { width: 100%; border-radius: 10px; border: 2px solid #e94560; margin-bottom: 15px; }
        input { width: 90%; padding: 12px; border-radius: 5px; border: none; margin-bottom: 15px; background: #0f3460; color: white; outline: none; }
        button { width: 100%; padding: 12px; background: #e94560; color: white; border: none; border-radius: 5px; font-weight: bold; cursor: pointer; transition: 0.3s; }
        button:hover { background: #c0354e; }
        #msg { margin-top: 15px; font-size: 0.9em; min-height: 20px; }
    </style>
</head>
<body>
    <div class="box">
        <h2>👁️ Privalens ID</h2>
        <video id="video" autoplay></video>
        <input type="text" id="username" placeholder="Enter Personnel Name" required autocomplete="off">
        <button id="snap">CAPTURE BIOMETRIC</button>
        <p id="msg"></p>
        <canvas id="canvas" width="320" height="240" style="display:none;"></canvas>
    </div>
    <script>
        const video = document.getElementById('video');
        const canvas = document.getElementById('canvas');
        const snap = document.getElementById('snap');
        const msg = document.getElementById('msg');
        const nameInput = document.getElementById('username');

        navigator.mediaDevices.getUserMedia({ video: true })
            .then(stream => { video.srcObject = stream; })
            .catch(err => { msg.innerText = "Error: Camera blocked"; msg.style.color = "red"; });

        snap.addEventListener("click", () => {
            const name = nameInput.value;
            if(!name) { 
                msg.innerText = "⚠️ Please enter a name first"; 
                msg.style.color = "yellow"; 
                return; 
            }
            
            // Visual feedback
            snap.innerText = "Processing...";
            snap.disabled = true;

            const context = canvas.getContext('2d');
            context.drawImage(video, 0, 0, 320, 240);
            const imageData = canvas.toDataURL('image/jpeg');

            fetch('/upload', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: name, image: imageData })
            })
            .then(res => res.json())
            .then(data => {
                msg.innerText = data.message;
                msg.style.color = data.message.includes("Success") ? "#4caf50" : "red";
                snap.innerText = "CAPTURE BIOMETRIC";
                snap.disabled = false;
                if(data.message.includes("Success")) nameInput.value = "";
            });
        });
    </script>
</body>
</html>