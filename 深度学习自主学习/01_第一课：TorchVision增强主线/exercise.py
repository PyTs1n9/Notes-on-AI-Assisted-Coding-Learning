import os
import torch
from torchvision import transforms as T
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, 'input')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

os.makedirs(OUTPUT_DIR, exist_ok=True)  # 函数③：自动创建输出目录

torch.manual_seed(42)
image_suffix = {'.jpg', '.jpeg', '.png', '.webp'}
input_files = []
for file_name in sorted(os.listdir(INPUT_DIR)):
	file_path = os.path.join(INPUT_DIR, file_name)
	if os.path.isfile(file_path) and os.path.splitext(file_name)[1].lower() in image_suffix:
		input_files.append(file_path)

transformer = T.Compose([
	T.Resize((160, 160)),
	T.RandomHorizontalFlip(p=1.0),
	T.ColorJitter(brightness=0.5),
	T.GaussianBlur(3, 0.5)
])

print('筛选后图片数量：', len(input_files))  # 显示筛选到的图片数
for file_path in input_files:  # 逐张处理图片
	with Image.open(file_path) as source_image:  # 打开图片并在退出时关闭文件
		image = source_image.convert('RGB')  # 函数⑫：统一成 RGB 图片
	transformed_image = transformer(image)  # 函数⑬：执行组合增强
	output_name = os.path.splitext(os.path.basename(file_path))[0] + '_aug.png'  # 函数⑭：生成新文件名
	output_path = os.path.join(OUTPUT_DIR, output_name)  # 函数②：拼出输出文件路径
	transformed_image.save(output_path)  # 函数⑮：保存增强图片
	print('已保存：', output_name)  # 显示输出文件名
	print('输出尺寸：', transformed_image.size)  # 显示图片尺寸
	print('输出模式：', transformed_image.mode)  # 显示颜色模式
