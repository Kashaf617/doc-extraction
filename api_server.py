import os
import sys
import logging
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from document_ai_system.pipeline_core import process_document
from document_ai_system.config import UPLOAD_DIR

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DocumentAI.API")

app = Flask(__name__, template_folder="templates")
CORS(app)

# Ensure upload directory exists
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf', 'tiff'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/process', methods=['POST'])
def process_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = str(UPLOAD_DIR / filename)
        file.save(filepath)
        
        try:
            logger.info(f"Processing uploaded file: {filepath}")
            result = process_document(filepath)
            
            # Clean up after processing (optional, but good for privacy/space)
            # os.remove(filepath)
            
            return jsonify(result)
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            return jsonify({"error": "Internal Processing Error", "message": str(e)}), 500
            
    return jsonify({"error": "File type not allowed"}), 400

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)

if __name__ == '__main__':
    # Use 0.0.0.0 to be accessible on local network if needed
    app.run(host='127.0.0.1', port=5000, debug=True)
