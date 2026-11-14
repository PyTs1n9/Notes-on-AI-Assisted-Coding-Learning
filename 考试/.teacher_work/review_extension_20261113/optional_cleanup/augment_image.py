import cv2


def fixed_rotate(img, angle=12):
	h, w = img.shape[:2]
	center = (w / 2, h / 2)
	martix = cv2.getRotationMatrix2D(center, angle, 1.0)
	img = cv2.warpAffine(
		img,
		martix,
		(w, h),
		flags=cv2.INTER_LINEAR,
		borderMode=cv2.BORDER_CONSTANT,
		borderValue=(0, 0, 0),
	)
	return img


def random_aug(img, rng):
	result = img.copy()
	flipped = False

	if rng.random() < 0.5:
		result = cv2.flip(result, 1)
		flipped = True

	angle = float(rng.uniform(-15, 15))
	result = fixed_rotate(result, angle)
	return result, flipped, angle
