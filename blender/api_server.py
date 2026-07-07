import os
import json
import subprocess
import tempfile
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)

# المسار إلى ملف بلندر التنفيذي عندك
BLENDER_PATH = r"C:\Program Files\Blender Foundation\Blender 4.1\blender.exe"

# مسار سكربت المعالجة الخاص ببلندر (الملف الثاني، منفصل تماماً)
BLENDER_WORKER_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "blender_worker.py"
)

ALLOWED_EXTENSIONS = {".fbx", ".glb", ".gltf"}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.path.join(SCRIPT_DIR, "dataset")
MODELS_FOLDER = os.path.join(DATASET_ROOT, "models")
GEOMETRY_ROOT = os.path.join(DATASET_ROOT, "geometry")

os.makedirs(MODELS_FOLDER, exist_ok=True)
os.makedirs(GEOMETRY_ROOT, exist_ok=True)

BLENDER_TIMEOUT_SECONDS = 600  # ارفعها لو عندك موديلات معقدة جداً


@app.route('/process-3d/', methods=['POST'])
def process_3d():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    uploaded_file = request.files['file']

    if uploaded_file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    # تنظيف اسم الملف لمنع path traversal
    safe_name = secure_filename(uploaded_file.filename)
    if not safe_name:
        return jsonify({"error": "Invalid filename"}), 400

    extension = os.path.splitext(safe_name)[1].lower()
    if extension not in ALLOWED_EXTENSIONS:
        return jsonify({
            "error": f"Unsupported file type '{extension}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}"
        }), 400

    # حفظ الملف
    file_path = os.path.join(MODELS_FOLDER, safe_name)
    uploaded_file.save(file_path)

    model_id = os.path.splitext(safe_name)[0]
    output_json_path = os.path.join(GEOMETRY_ROOT, f"{model_id}.json")

    # حذف نتيجة قديمة إن وجدت، حتى لا نقرأ بيانات قديمة لو بلندر فشل
    if os.path.exists(output_json_path):
        os.remove(output_json_path)

    try:
        result = subprocess.run(
            [
                BLENDER_PATH,
                "--background",
                "--python", BLENDER_WORKER_SCRIPT,
                "--", file_path
            ],
            capture_output=True,
            text=True,
            timeout=BLENDER_TIMEOUT_SECONDS
        )

        if not os.path.exists(output_json_path):
            return jsonify({
                "error": "Blender script failed to generate geometry JSON",
                "blender_returncode": result.returncode,
                # تقليم الإخراج الطويل
                "blender_stderr": result.stderr[-4000:],
                "blender_stdout": result.stdout[-4000:]
            }), 500

        with open(output_json_path, "r", encoding="utf8") as f:
            blender_calculated_data = json.load(f)

        # حقن تصنيف AI وهمي ريثما تربط موديل التصنيف الحقيقي
        blender_calculated_data["ai_classification"] = {
            "label": "Chair",
            "confidence": 0.95
        }

        return jsonify(blender_calculated_data), 200

    except subprocess.TimeoutExpired:
        return jsonify({
            "error": f"Blender process timed out after {BLENDER_TIMEOUT_SECONDS}s"
        }), 504

    except FileNotFoundError:
        return jsonify({
            "error": f"Blender executable not found at: {BLENDER_PATH}"
        }), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        # تنظيف الملف المرفوع بعد المعالجة لتفادي تراكم البيانات على القرص
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError:
                pass


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
