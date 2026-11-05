import os
import cv2
import numpy as np
from torch._C import TensorType


def Scale(image, scale):
	return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)


def Horizontal(image):
	return cv2.flip(image, 1, dst=None)


def Vertical(image):
	return cv2.flip(image, 0, dst=None)


def Rotate(image, angle=15, scale=0.9):
	w = image.shape[1]
	h = image.shape[0]
	M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
	image = cv2.warpAffine(image, M, (w, h))
	return image


def TestPic():
	test_jpg_loc = r"./img/image_000.png"
	test_jpg = cv2.imread(test_jpg_loc)
	cv2.imshow('xxx', test_jpg)
	cv2.waitKey(0)

	img1 = Scale(test_jpg, 0.5)
	cv2.imshow("img1", img1)
	cv2.waitKey(0)
	img2 = Scale(test_jpg, 2)
	cv2.imshow("img2", img2)
	cv2.waitKey(0)


if __name__ == '__main__':
	TestPic()
