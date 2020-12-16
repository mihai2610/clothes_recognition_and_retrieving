import torch
import os
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from PIL import Image


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
	transforms = []
	transforms.append(ToTensor())
	return Compose(transforms)


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
			result[categories[labels[idx]]] = scores[idx]

		return result

# if __name__ == '__main__':
# 	model = ClothesDetectionModel(r"D:\master\an2\SRI\project\clothes_recognition_and_retrieving")
# 	path = os.path.join(r"D:\master\an2\SRI\project\clothes_recognition_and_retrieving\data\deepfashion\train\image", "000011.jpg")
# 	print(model.get_image_label(path))
