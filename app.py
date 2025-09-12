import os
import sys
import glob
import subprocess
from pathlib import Path
from flask import Flask, render_template, request, jsonify, url_for, redirect, send_from_directory, session, abort
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import markdown2
from readme_parser import convert_markdown_to_sections

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-key-change-me")

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
ALLOWED_EXT = {".csv"}

def _allowed_file(filename: str) -> bool:
    return "." in filename and Path(filename).suffix.lower() in ALLOWED_EXT

@app.get("/")
def index():
    return render_template("index.html")

@app.post("/upload")
def upload():
    f = request.files.get("file")
    if not f or f.filename == "":
        return jsonify(ok=False, error="No file provided."), 400
    if not _allowed_file(f.filename):
        return jsonify(ok=False, error="Only .csv files are allowed."), 400
    
    safe_name = secure_filename(f.filename)
    dest = UPLOAD_DIR / safe_name
    f.save(dest)

    session["uploaded_path"] = str(dest)
    return jsonify(ok=True, filename=safe_name)

@app.post("/generate")
def generate():
    uploaded_path = session.get("uploaded_path")
    if not uploaded_path:
        return jsonify(ok=False, error="No uploaded file found in session."), 400
    
    csv_path = Path(uploaded_path)
    dataset_name = csv_path.stem
    try:
        subprocess.run(
            [sys.executable, str(BASE_DIR / "autolysis.py"), str(csv_path)],
            cwd=str(BASE_DIR),
            check=True,
            capture_output=True,
            text=True,
            )
    except subprocess.CalledProcessError as e:
        return jsonify(ok=False, error=f"autolysis failed: {e.stderr or e.stdout}"), 500
    
    # clear the session value so a new upload is required next time
    session.pop("uploaded_path", None)
    return jsonify(ok=True, redirect=url_for("report", dataset=dataset_name))

@app.get("/report/<dataset>")
def report(dataset: str):
    # allow alnum, dash, underscore
    if not all(c.isalnum() or c in {"-", "_"} for c in dataset):
        abort(404)

    folder = BASE_DIR / dataset
    if not folder.is_dir():
        abort(404)

    readme_md = (folder / "README.md").read_text(encoding="utf-8") if (folder / "README.md").exists() else "# README.md not found"
    sections = convert_markdown_to_sections(readme_md)
    
    image_paths = sorted(glob.glob(str(folder / "*.png")))
    image_map = {}
    for p in image_paths:
        title = Path(p).stem.replace("_", " ").title()
        url = url_for("report_file", dataset=dataset, filename=Path(p).name)
        key = title.lower().strip()
        image_map[key] = url
        # add common aliases so that headings in README always match
        if "correlation heatmap" in key and "matrix" not in key:
            image_map["correlation matrix heatmap"] = url
        if "correlation matrix heatmap" in key:
            image_map["correlation heatmap"] = url

    return render_template("report.html", dataset=dataset, sections=sections, image_map=image_map)

@app.get("/report/<dataset>/files/<path:filename>")
def report_file(dataset: str, filename: str):
    # Only expose README.md and PNG assets from the dataset folder
    if not (filename == "README.md" or Path(filename).suffix.lower() == ".png"):
        abort(404)
    folder = BASE_DIR / dataset
    return send_from_directory(str(folder), filename, as_attachment=False)

if __name__ == "__main__":
    app.run(debug=True)