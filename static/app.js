// JS to handle drag&drop upload and the loading overlay during generation
(function () {
  const drop = document.getElementById('dropzone');
  const input = document.getElementById('file-input');
  const statusEl = document.getElementById('status');
  const genBtn = document.getElementById('generate-btn');
  const overlay = document.getElementById('loadingOverlay');

  function setLoaded(ok) {
    if (ok) {
      statusEl.classList.remove('d-none');
      genBtn.disabled = false;
    } else {
      statusEl.classList.add('d-none');
      genBtn.disabled = true;
    }
  }

  function showOverlay(msg) {
    overlay.querySelector('.msg').textContent = msg || "Please wait. Report is getting generated...";
    overlay.style.display = 'flex';
  }
  function hideOverlay() { overlay.style.display = 'none'; }

  async function upload(file) {
    const fd = new FormData();
    fd.append('file', file);
    const r = await fetch('/upload', { method: 'POST', body: fd });
    const data = await r.json();
    if (!data.ok) throw new Error(data.error || 'Upload failed');
    setLoaded(true);
  }

  // Input picker
  input.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      upload(e.target.files[0]).catch(err => alert(err.message));
    }
  });

  // Drag & drop
  ['dragenter','dragover'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add('dragover'); }));
  ['dragleave','drop'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove('dragover'); }));
  drop.addEventListener('drop', (e) => {
    const f = e.dataTransfer.files?.[0];
    if (f) upload(f).catch(err => alert(err.message));
  });

  // Generate
  genBtn.addEventListener('click', async () => {
    try{
      showOverlay("Please wait. Report is getting generated...");
      const r = await fetch('/generate', { method: 'POST' });
      const data = await r.json();
      if (!data.ok) throw new Error(data.error || 'Generation failed');
      window.location.href = data.redirect; // overlay stays until navigation
    }catch(err){
      hideOverlay();
      alert(err.message);
    }
  });

  // Start disabled
  setLoaded(false);
})();
