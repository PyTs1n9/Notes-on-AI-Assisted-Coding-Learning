import os
import hashlib
from torchvision import transforms as T
import torch
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, 'input')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.webp'}
BASE_SEED = 73
os.makedirs(OUTPUT_DIR, exist_ok=True)
transform = T.Compose([
	T.Resize((160, 160)),
	T.RandomHorizontalFlip(p=0.5),
	T.RandomRotation(degrees=20, fill=(0, 0, 0)),
	T.ColorJitter(brightness=0.3),
	T.GaussianBlur(kernel_size=3, sigma=0.5)
])


def get_image_files(input_dir):
	image_paths = []
	for file_name in sorted(os.listdir(input_dir)):
		file_path = os.path.join(input_dir, file_name)
		suffix = os.path.splitext(file_name)[1].lower()
		if os.path.isfile(file_path) and suffix in IMAGE_SUFFIXES:
			image_paths.append(file_path)
		else:
			print('skip', file_name)
	return image_paths


def make_output_path(file_path, output_dir):  # 定义函数：创建不冲突的输出文件路径
	base_name = os.path.splitext(os.path.basename(file_path))[0]  # 函数⑤：取输入图片的主文件名
	output_name = base_name + '_aug.png'  # 给增强结果添加后缀
	output_path = os.path.join(output_dir, output_name)  # 函数②：拼接输出路径
	number = 1  # 设置冲突文件的起始编号
	while os.path.exists(output_path):  # 函数⑨：检查目标文件是否已经存在
		output_name = base_name + '_aug_' + str(number) + '.png'  # 生成带编号的新文件名
		output_path = os.path.join(output_dir, output_name)  # 函数②：更新目标路径
		number = number + 1  # 编号加一，继续检查下一个名称
	return output_path  # 返回不会覆盖已有文件的路径


def enhance_one_image(file_path, output_dir, image_index):
	torch.manual_seed(BASE_SEED + image_index)
	with Image.open(file_path) as source_image:
		image = source_image.convert('RGB')
	enhanced_image = transform(image)
	output_path = make_output_path(file_path, output_dir)
	enhanced_image.save(output_path)
	return output_path, enhanced_image.size, enhanced_image.mode


image_paths = get_image_files(INPUT_DIR)  # 收集输入目录中的图片
print('筛选后图片数量：', len(image_paths))  # 输出筛选数量
for image_index in range(len(image_paths)):
	output_path, output_size, output_mode = enhance_one_image(
		image_paths[image_index], OUTPUT_DIR, image_index
	)
	print('已保存：', os.path.basename(output_path))  # 输出保存后的文件名
	print('输出尺寸：', output_size)  # 输出图像实际尺寸
	print('输出模式：', output_mode)  # 输出图像颜色模式
	with open(output_path, 'rb') as image_file:  # 以二进制只读方式打开生成图片
		image_hash = hashlib.sha256(image_file.read()).hexdigest()  # 函数⑨：计算图片SHA-256
	print('SHA-256：', image_hash)  # 输出图片的字节指纹
