from flask import Flask, jsonify, json, request, Response, render_template, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
import os
from datetime import datetime
from model.model import ClothesDetectionModel
import json

category_to_id = {
	"short sleeve top": 1,
	"long sleeve top": 2,
	"short sleeve outwear": 4,
	"long sleeve outwear": 4,
	"vest": 4,
	"sling": 6,
	"shorts": 8,
	"trousers": 8,
	"skirt": 9,
	"short sleeve dress": 10,
	"long sleeve dress": 10,
	"vest dress": 10,
	"sling dress": 10
}


def get_recommendation(predicted_items, max_items=3, max_pictures=3):
	recommendations = []

	for pred in predicted_items:
		id = category_to_id[pred]
		recommendations_count = max_items

		items = {}
		with open("filtered_clothes.json", 'r') as outfile:
			items = json.load(outfile)

		for item in items:
			if item["category_id"] == id:
				_item = item["images"][0]
				_page = item["page"]

				recommendations.append({"image": _item, "url": _page})

				recommendations_count -= 1

			if recommendations_count <= 0:
				break

	return recommendations


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

		return jsonify({'filename': filename})

	return render_template('home.html', response1=False, upload_image=None)


@app.route("/", methods=['GET'])
def main():
	filename = request.args.get('filename', '')
	upload_image = None
	items = []
	if filename:
		file_path = 'uploaded_images/' + filename
		upload_image = url_for('static', filename=file_path)

		img_path = os.path.join('static', file_path)

		# return a dictionary with labels and scores
		results = detection_model.get_image_label(os.path.join('static', file_path))

		# return the dominant color of the first box
		color = detection_model.dominant_box_color(img_path)

		print(results, color)

		_results = [res for res in results.keys() if results[res] > 0.7]
		print(_results)

		items = get_recommendation(_results)

	return render_template('home.html', items=items, upload_image=upload_image)


if __name__ == "__main__":
	app.run(host='127.0.0.1', port=8001, debug=True)
