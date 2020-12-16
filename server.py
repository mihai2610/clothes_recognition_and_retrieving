from flask import Flask, jsonify, json, request, Response, render_template, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
import os
from datetime import datetime
from .model.model import  ClothesDetectionModel

app = Flask(__name__)
detection_model = ClothesDetectionModel(r"D:\master\an2\SRI\project\clothes_recognition_and_retrieving")

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('file', '')

    if file:
        filename = secure_filename(file.filename)
        filename, extension = os.path.splitext(filename)
        filename = filename + datetime.now().strftime("%d_%m_%Y %H_%M_%S") + extension

        file_path = os.path.join('static/uploaded_images', filename)

        file.save(file_path)

        detection_model.get_image_label(file_path)

        return jsonify({'filename': filename})

    return render_template('home.html', response1=False, upload_image=None)


@app.route("/", methods=['GET'])
def main():
    filename = request.args.get('filename', '')
    items = None
    upload_image = None

    if filename:
        upload_image = url_for('static', filename='uploaded_images/' + filename)
        items = [
            {
                'image': 'https://images-na.ssl-images-amazon.com/images/I/610irNyucGL._UL1500_.jpg',
                'url': 'https://www.cleany.ro',
            },
            {
                'image': 'https://www.sunspel.com/media/catalog/product/cache/bee6a030eca197d5b3b98b85dbca461b/m/t/mtsh0001-gyab_1_1.jpg',
                'url': 'https://www.cleany.ro',
            }
        ]

    return render_template('home.html', items=items, upload_image=upload_image)


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8001, debug=True)