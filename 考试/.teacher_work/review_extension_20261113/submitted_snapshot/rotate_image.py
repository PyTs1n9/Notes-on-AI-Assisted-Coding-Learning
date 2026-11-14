import cv2


def fixed_rotate(img, angle=12):
	h, w = img.shape[:2]
	center = (w / 2, h / 2)
	martix = cv2.getRotationMatrix2D(center, angle, 1.0)
	img = cv2.warpAffine(img, martix, (w, h))
	return img
