from deepface import DeepFace
import os
import zipfile
from flask import Flask, request, jsonify

app = Flask(__name__)

# Paths and model name
archive_path = "archive.zip"
db_extracted_path = r"C:/Users/tragu/Downloads/archive_extracted"
model_name = "ArcFace"

def unzip_archive(archive_path, extract_to):
    with zipfile.ZipFile(archive_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def recognition(image_path, db_path):
    if not os.path.exists(db_path):
        unzip_archive(archive_path, db_extracted_path)
    dfs = DeepFace.find(img_path=image_path, db_path=db_extracted_path, model_name=model_name)
    return dfs

# Define Flask endpoints
@app.route('/face_recognition', methods=['POST'])
def face_recognition():
    image1 = request.files['image']
    image_path = os.path.join(db_extracted_path, image1.filename)
    image1.save(image_path)
    faces = recognition(image_path, db_extracted_path)
    return jsonify({'faces': faces})

@app.route('/test')
def test():
    return "success"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

