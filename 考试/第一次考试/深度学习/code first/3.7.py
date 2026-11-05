# 7-1

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from PIL import Image
from torchvision import transforms
from matplotlib import rcParams
import matplotlib.pyplot as plt

input_dir = r'../increase_data'
output_dir = r'../increase_result'
img_suffix = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

count = 0
for img in os.listdir(input_dir):
	full_path = os.path.join(input_dir, img)
	if img.lower().endswith(img_suffix):
		count = count + 1
		with Image.open(full_path) as image:
			image = image.convert('RGB')
print(count)

# 7-2


augment = transforms.Compose([
	transforms.RandomHorizontalFlip(p=0.5),
	transforms.RandomRotation(degrees=15),
	transforms.ColorJitter(
		brightness=0.2,
		contrast=0.2,
		saturation=0.2
	)
])

# 7-3
for img in os.listdir(input_dir):
	if img.lower().endswith(img_suffix):
		input_path = os.path.join(input_dir, img)
		with Image.open(input_path) as image:
			image = image.convert('RGB')

			name, _ = os.path.splitext(img)
			for index in range(1, 4):
				augment_img = augment(image)
				output_path = os.path.join(
					output_dir, f'{name}_aug_{index}.jpg'
				)
				augment_img.save(output_path)

# 7.4
original_path = os.path.join(input_dir, os.listdir(input_dir)[0])
name, _ = os.path.splitext(os.path.basename(original_path))
augmented_path = os.path.join(output_dir, f'{name}_aug_1.jpg')

with Image.open(original_path) as image:
	original_image = image.convert("RGB")

with Image.open(augmented_path) as image:
	augmented_image = image.convert("RGB")

rcParams['font.sans-serif'] = ['SimHei']
plt.subplot(1, 2, 1)
plt.imshow(original_image)
plt.title("原图")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(augmented_image)
plt.title("增强后")
plt.axis("off")
plt.show()

print("原始图片数量：", count)
print("增强后图片数量：", count * 3)
print("增强方法：随机水平翻转（增加左右方向变化）、随机旋转（增加角度变化）、亮度/对比度/饱和度调整（增加颜色变化）。")
print("增强图片保存在 increase_result，原始图片保留在 increase_data，未被覆盖或修改。")
