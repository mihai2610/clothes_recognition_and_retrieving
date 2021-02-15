import torch
import os
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from PIL import Image
import sys
import cv2
from sklearn.cluster import KMeans
import numpy as np
from collections import Counter

from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import json


class ToTensor(object):
	def __call__(self, image):
		image = F.to_tensor(image)
		return image


class Compose(object):
	def __init__(self, transforms):
		self.transforms = transforms

	def __call__(self, image):
		for t in self.transforms:
			image = t(image)
		return image


def get_transform():
	return Compose([ToTensor()])


num_classes = 13

categories = {
	1: "short sleeve top",
	2: "long sleeve top",
	3: "short sleeve outwear",
	4: "long sleeve outwear",
	5: "vest",
	6: "sling",
	7: "shorts",
	8: "trousers",
	9: "skirt",
	10: "short sleeve dress",
	11: "long sleeve dress",
	12: "vest dress",
	0: "sling dress"
}


class ClothesDetectionModel(object):

	def __init__(self, base_path=""):
		self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
		in_features = self.model.roi_heads.box_predictor.cls_score.in_features
		self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
		self.model.load_state_dict(torch.load(os.path.join(base_path, r"model/state_dict_model_1.pt")))

		self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

		self.model.to(self.device)

		self.activation = {}

		self.n_clusters = 3

	def get_activation(self, name):
		def hook(model, input, output):
			if name not in self.activation.keys():
				self.activation[name] = [output.detach()]
			else:
				self.activation[name].append(output.detach())

		# print("activation name", self.activation[name])
		# print("activation", self.activation)

		return hook

	def dominant_colors(self, img):
		# convert to rgb from bgr
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

		# reshaping to a list of pixels
		img = img.reshape((img.shape[0] * img.shape[1], 3))

		# using k-means to cluster pixels
		kmeans = KMeans(n_clusters=self.n_clusters)
		kmeans.fit(img)

		# the cluster centers are our dominant colors.
		COLORS = kmeans.cluster_centers_

		count = Counter(kmeans.labels_)

		max_cluster = max(count, key=lambda k: count[k])

		# returning after converting to integer from float
		# print(COLORS.astype(int))
		# print("dist", self.color_dist(COLORS.astype(int)[0], COLORS.astype(int)[1]))
		# print("best:", max_cluster)

		return COLORS.astype(int)[max_cluster]

	def dominant_box_color(self, img_path, bad_converts=None):
		img = cv2.imread(img_path)
		boxes = self.get_image_boxes(img)

		# for box_key in boxes:
		# 	box = boxes[box_key].astype(int)

		if len(boxes) < 1:
			if bad_converts:
				bad_converts[0] += 1
			return []

		# cv2.imshow('image', _img)
		# cv2.waitKey(0)

		box = boxes[list(boxes.keys())[0]].astype(int)

		if img.shape[0] <= box[0] or img.shape[1] <= box[1]:
			if bad_converts:
				bad_converts[0] += 1
			return []
		else:
			_img = img[box[0]: box[2], box[1]: box[3]]

		color = [int(color) for color in self.dominant_colors(_img)]

		return color

	def color_dist(self, color1, color2):
		return np.sqrt(pow(color1[0] - color2[0], 2) + pow(color1[1] - color2[1], 2) + pow(color1[2] - color2[2], 2))

	def get_image_label(self, img_path):
		self.model.eval()
		result = dict()

		img = Image.open(img_path).convert("RGB")
		img = F.to_tensor(img)

		with torch.no_grad():
			prediction = self.model([img.to(self.device)])

		labels = prediction[0]['labels'].cpu().numpy()
		scores = prediction[0]['scores'].cpu().numpy()

		for idx in range(3):
			if idx >= len(labels):
				break
			result[categories[labels[idx]]] = scores[idx]

		return result

	def get_image_boxes(self, image):
		self.model.eval()
		result = dict()

		img = F.to_tensor(image)

		with torch.no_grad():
			prediction = self.model([img.to(self.device)])

		labels = prediction[0]['labels'].cpu().numpy()
		boxes = prediction[0]['boxes'].cpu().numpy()
		scores = prediction[0]['scores'].cpu().numpy()

		for idx in range(3):
			if idx >= len(scores):
				break
			if scores[idx] > 0.7:
				result[categories[labels[idx]]] = boxes[idx]

		return result

	def vector_distance(self, x1, x2):
		return np.linalg.norm(np.asarray(x1) - np.asarray(x2))

	def get_vector(self):
		return torch.sum(self.activation['fc6'][-1], dim=0).tolist()

	def init_hook(self):
		i = 0
		for child in self.model.roi_heads.children():
			if i == 1:
				for layer in child.children():
					layer.register_forward_hook(self.get_activation('fc6'))
					break
			i += 1


def save_image_vector():
	detection_model = ClothesDetectionModel(r"D:\master\an2\SRI\project\clothes_recognition_and_retrieving")
	detection_model.init_hook()

	with open("data/all_clothes.json", 'r') as outfile:
		items = json.load(outfile)

	index = 0
	for item in items:
		index += 1
		if index <= 1000:
			continue

		image_path = r"" + item['local_images'][0]
		if os.path.exists(image_path):
			detection_model.get_image_label(image_path)
			item['vector'] = torch.sum(detection_model.activation['fc6'][-1], dim=0).tolist()

		if index % 10 == 0:
			print(index)

	with open("data/all_clothes.json", 'w') as outfile:
		json.dump(items, outfile)

# save_image_vector()

# if __name__ == '__main__':
# 	path = os.path.join(r"D:\master\an2\SRI\project\clothes_recognition_and_retrieving\data\deepfashion\train\image", "000011.jpg")
# 	print(model.get_image_label(path))
