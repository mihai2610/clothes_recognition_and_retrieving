from flask import Flask, jsonify, json, request, Response, render_template, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
import os
from datetime import datetime
from model.model import ClothesDetectionModel, save_image_vector
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


app = Flask(__name__)
detection_model = ClothesDetectionModel(r"D:\master\an2\SRI\project\clothes_recognition_and_retrieving")
detection_model.init_hook()

def get_recommendation(predicted_items, main_color, main_vector, is_color, max_items=10):
	recommendations = []

	items = {}
	with open("data/all_clothes.json", 'r') as outfile:
		items = json.load(outfile)

	dists = []
	print(category_to_id[predicted_items[0]])
	for item in items:
		if item["category_id"] == category_to_id[predicted_items[0]]:
			_item = item["images"][0]
			_page = item["page"]

			if is_color:
				if len(item['rgb_color']) != 3:
					continue
				dists.append(detection_model.color_dist(main_color, item['rgb_color']))
			else:
				if 'vector' not in item.keys():
					continue
				dists.append(detection_model.vector_distance(main_vector, item['vector']))

			recommendations.append({"image": _item, "url": _page, "price": str(item['lei']) + " RON"})

	top_recommandations = []
	current_recomm_vector = dists
	while max_items:
		min_dist = min(current_recomm_vector)
		print(min_dist)
		index = current_recomm_vector.index(min_dist)

		top_recommandations.append(recommendations[index])
		recommendations.pop(index)
		current_recomm_vector.pop(index)

		max_items -= 1

	print(top_recommandations)
	return top_recommandations



def get_files_name(dir_name='data/images'):
	paths = []
	for path in os.listdir(dir_name):
		paths.append(dir_name + "/" + path)

	return paths


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
	is_color = request.args.get('is_color', True)
	if is_color == 'false':
		is_color = False

	upload_image = None
	items = None
	if filename:
		file_path = 'uploaded_images/' + filename
		upload_image = url_for('static', filename=file_path)

		img_path = os.path.join('static', file_path)

		# return a dictionary with labels and scores
		results = detection_model.get_image_label(os.path.join('static', file_path))
		vector = detection_model.get_vector()

		max_score = max(results.values())
		if max_score <= 0.7:
			results = []
		else:
			key = list(results.keys())[list(results.values()).index(max_score)]
			results = [key]

		# return the dominant color of the first box
		main_color = detection_model.dominant_box_color(img_path)

		items = get_recommendation(results, main_color, vector, is_color)

	return render_template('home.html', items=items, upload_image=upload_image)


def save_image_color():
	with open("data/all_clothes.json", 'r') as outfile:
		items = json.load(outfile)

	index = 0
	bad_values = [0]
	for item in items:
		if index % 10 == 0:
			print(index)
		index += 1

		image_path = r"" + item['local_images'][0]
		if os.path.exists(image_path):
			item['rgb_color'] = list(detection_model.dominant_box_color(image_path, bad_values))
		else:
			print("no_image")

	print("bad_values", bad_values)

	with open("data/all_clothes.json", 'w') as outfile:
		json.dump(items, outfile)


if __name__ == "__main__":
	app.run(host='127.0.0.1', port=8001, debug=True)

