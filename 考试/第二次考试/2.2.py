import cv2
import numpy as np
import random
import os


# img = os.path.abspath(__file__)
# base = os.path.dirname(img)
# input_img = os.path.join(base, 'out.png')
# imgs = './out.png'
# image = Image.open(imgs).convert('RGB')

# def random_augment(img):4
# 	# 使用torchvision实现随机增强组合函数，要定义def
# 	transform = T.Compose([
# 		T.RandomHorizontalFlip(p=0.5),
# 		T.RandomRotation([-15, 15]),
# 		T.GaussianBlur(1, 0.1),
# 		T.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
# 		T.RandomGrayscale(p=0.2)
# 	])
# 	return transform(img)

def random_augment(img):
	if random.random() < 0.5:
		img = cv2.flip(img, 1)
	if random.random() < 0.5:
		h, w = img.shape[:2]
		M = cv2.getRotationMatrix2D((w / 2, h / 2), random.uniform(-15, 15), 1.0)
		img = cv2.warpAffine(img, M, dsize=(w, h))
	if random.random() < 0.5:
		noise = np.random.normal(0, 10, img.shape)
		img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
	return img
