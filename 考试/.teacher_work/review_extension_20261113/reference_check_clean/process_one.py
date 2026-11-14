import os, cv2

import numpy as np

base = os.path.dirname(os.path.abspath(__file__))
# 输入目录
input_path = os.path.join(base, 'input/input.png')
# 输出目录
os.makedirs('output_supplement', exist_ok=True)
output_dir = os.path.join(base, 'output_supplement')
# 后缀
extensions = ('.png', '.jpg', '.jpeg', '.bmp')


def read_image(path):
	data = np.fromfile(path, dtype=np.uint8)
	if data.size == 0:
		return None
	return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_image(path, image):
	ext = os.path.splitext(path)[1].lower()
	success, encoded = cv2.imencode(ext, image)
	if not success:
		return False
	encoded.tofile(path)
	return os.path.isfile(path)


def rotate_image(img, angle):
	h, w = img.shape[:2]
	center = (w / 2, h / 2)
	martix = cv2.getRotationMatrix2D(center, angle, 1.0)
	rotated = cv2.warpAffine(
		img,
		martix,
		(w, h),
		flags=cv2.INTER_LINEAR,
		borderMode=cv2.BORDER_CONSTANT,
		borderValue=(0, 0, 0)
	)
	return rotated


def augment_image(img, rng):
	result = img.copy()
	flipped = False
	if rng.random() < 0.5:
		result = cv2.flip(result, 1)
		flipped = True
	angle = float(rng.uniform(-15, 15))
	result = rotate_image(result, angle)
	return result, flipped, angle


def process_one(input_dir, filename, output_dir):
	ext = os.path.splitext(filename)[1].lower()
	if ext not in extensions:
		return False, filename + '不支持扩展名'

	input_path = os.path.join(input_dir, filename)
	img = read_image(input_path)
	if img is None:
		return False, filename + '读取失败'

	base_name = os.path.splitext(filename)[0]
	flip_path = os.path.join(output_dir, base_name + '_flip.png')
	rotate_path = os.path.join(output_dir, base_name + '_rotate.png')
	if os.path.exists(flip_path) or os.path.exists(rotate_path):
		return False, filename + '输出已经存在，拒绝覆盖'

	flip_result = cv2.flip(img, 1)
	rotate_result = rotate_image(img, 12)
	if not write_image(flip_path, flip_result):
		return False, filename + '翻转结果保存失败'
	if not write_image(rotate_path, rotate_result):
		return False, filename + '旋转结果保存失败'

	return True, filename + '已生成flip和rotate'


def batch_augment(input_dir, output_dir):
	os.makedirs(output_dir, exist_ok=True)
	success_count = 0
	failure_count = 0
	messages = []
	for filename in sorted(os.listdir(input_dir)):
		path = os.path.join(input_dir, filename)
		extension = os.path.splitext(filename)[1].lower()
		if not os.path.isfile(path) or extension not in extensions:
			continue
		success, message = process_one(input_dir, filename, output_dir)
		messages.append(message)
		if success:
			success_count += 1
		else:
			failure_count += 1
	return success_count, failure_count, messages


def main():
	img = read_image(input_path)
	print("input_exists", os.path.isfile(input_path))
	print("input_shape", img.shape)
	print("input_dtype", img.dtype)

	fixed = rotate_image(img, 12)
	print("fixed_shape", fixed.shape)
	print("fixed_dtype", fixed.dtype)
	print("fixed_changed", not np.array_equal(img, fixed))

	rng_a = np.random.default_rng(42)
	rng_b = np.random.default_rng(42)
	augmented_a, flipped_a, angle_a = augment_image(img, rng_a)
	augmented_b, flipped_b, angle_b = augment_image(img, rng_b)
	print("augment_reproducible", np.array_equal(augmented_a, augmented_b))
	print("augment_same_decision", flipped_a == flipped_b)
	print("augment_same_angle", angle_a == angle_b)
	print("augment_shape", augmented_a.shape)
	print("augment_dtype", augmented_a.dtype)
	print("input_unchanged", np.array_equal(img, read_image(input_path)))

	success, failure, messages = batch_augment(os.path.dirname(input_path), output_dir)
	print("batch_first", success, failure)
	for message in messages:
		print("message", message)

	output_names = sorted(os.listdir(output_dir))
	print("output_names", output_names)
	print("output_count", len(output_names))
	for name in output_names:
		output = read_image(os.path.join(output_dir, name))
		print("artifact", name, output.shape, output.dtype, output is not None)
	flip = read_image(os.path.join(output_dir, "input_flip.png"))
	rotate = read_image(os.path.join(output_dir, "input_rotate.png"))
	print("flip_matches", np.array_equal(flip, cv2.flip(img, 1)))
	print("rotate_changed", not np.array_equal(rotate, img))

	success, failure, messages = batch_augment(os.path.dirname(input_path), output_dir)
	print("batch_second", success, failure)
	for message in messages:
		print("message", message)
	print("output_count_after_second", len(os.listdir(output_dir)))


if __name__ == "__main__":
	main()