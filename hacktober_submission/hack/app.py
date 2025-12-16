import os
import json
import time
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from model.fashion_model import evaluate_outfit

# ---- Setup ----
app = Flask(__name__, static_folder="static", static_url_path="/static")
UPLOAD_FOLDER = "uploads"
DATA_FILE = "outfits.json"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXT = {"png", "jpg", "jpeg"}

# Add CORS headers
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

def load_db():
    if not os.path.exists(DATA_FILE):
        return []
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_db(data):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

# ---- Routes ----
@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# ---------- 1️⃣ Upload Section ----------
@app.route("/upload", methods=["POST"])
def upload_image():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400
        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type"}), 400

        filename = f"{int(time.time()*1000)}_{secure_filename(file.filename)}"
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(save_path)

        db = load_db()
        db.append({
            "filename": filename,
            "url": f"/uploads/{filename}",
            "timestamp": time.strftime("%d/%m/%Y, %I:%M:%S %p")
        })
        save_db(db)

        return jsonify({"message": "Uploaded successfully!"})
    
    except Exception as e:
        print(f"Upload error: {str(e)}")
        return jsonify({"error": f"Upload failed: {str(e)}"}), 500

@app.route("/gallery", methods=["GET"])
def gallery():
    db = load_db()
    return jsonify({"outfits": db})

# ---------- 2️⃣ Generate Outfit Section ----------
@app.route("/generate", methods=["POST"])
def generate_best():
    data = request.json
    occasion = data.get("occasion")
    style = data.get("style")

    if not occasion or not style:
        return jsonify({"error": "Please select occasion and style"}), 400

    db = load_db()
    if not db:
        return jsonify({"error": "No outfits uploaded yet"}), 400

    results = []
    for outfit in db:
        img_path = os.path.join(UPLOAD_FOLDER, outfit["filename"])
        try:
            score = evaluate_outfit(img_path, occasion.lower(), style.lower())
        except Exception as e:
            score = 0.0
        outfit["score"] = round(float(score), 4)
        outfit["occasion"] = occasion
        outfit["style"] = style
        results.append(outfit)

    results = sorted(results, key=lambda x: x["score"], reverse=True)
    save_db(results)
    best = results[0] if results else None

    return jsonify({
        "message": "Evaluation complete",
        "best": best,
        "outfits": results
    })

# ---------- 3️⃣ Reset ----------
@app.route("/reset", methods=["POST"])
def reset():
    db = load_db()
    for r in db:
        path = os.path.join(UPLOAD_FOLDER, r["filename"])
        if os.path.exists(path):
            os.remove(path)
    if os.path.exists(DATA_FILE):
        os.remove(DATA_FILE)
    return jsonify({"message": "All outfits cleared."})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

