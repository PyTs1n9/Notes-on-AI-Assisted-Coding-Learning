# import cv2
# import os
#
# img = os.path.abspath(__file__)
# base = os.path.dirname(img)
# input_img = os.path.join(base,'out.png')
# imgs = './out.png'
#
# def random_rotate(img, max_angle=15):
# 	"""
# 	第一个参数是旋转中心点，第二个参数是旋转角度，第三个参数是保持尺寸与原图一致
# 	"""
# 	rot_matrix = cv2.getRotationMatrix2D((100, 100), max_angle, 1)
#
# 	# 使用旋转矩阵将图像旋转
# 	image = cv2.warpAffine(img, rot_matrix, (img.shape[1], img.shape[0]))
# 	return image
#
# random_rotate(imgs)

import os
import random
import cv2
import numpy as np


def random_rotate(img, max_angle=15):
	"""getRotationMatrix2D(center, angle, scale)：center 为旋转中心 (x, y)；angle 为旋转角度（度，正值逆时针）；scale 为等比缩放因子"""
	h, w = img.shape[:2]
	random_angle = random.uniform(-max_angle, max_angle)
	W = cv2.getRotationMatrix2D((w / 2, h / 2), random_angle, 1.0)
	return cv2.warpAffine(img,W,dsize=(w,h))


base = os.path.dirname(os.path.abspath(__file__))
src = os.path.join(base, 'out.png')
dst = os.path.join(base, 'output', 'out_rotate.png')

img = cv2.imdecode(np.fromfile(src, dtype=np.uint8), cv2.IMREAD_COLOR)
rot = random_rotate(img)
ok, buf = cv2.imencode('.png', rot)
if ok:
	buf.tofile(dst)
print(dst)
