import os  # 导入 os，用于处理文件路径、目录和文件遍历
import random  # 导入 random，用于固定 Python 随机数
import torch  # 导入 torch，用于固定 PyTorch 随机数
from PIL import Image, ImageDraw  # 导入图片对象和绘图工具
from torchvision import transforms  # 导入 TorchVision 的图像增强工具

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 函数①：获取本课文件所在目录
INPUT_DIR = os.path.join(BASE_DIR, "input")  # 函数②：拼接输入图片目录
OUTPUT_DIR = os.path.join(BASE_DIR, "output")  # 函数③：拼接输出图片目录
os.makedirs(INPUT_DIR, exist_ok=True)  # 函数④：输入目录不存在时创建目录
os.makedirs(OUTPUT_DIR, exist_ok=True)  # 函数⑤：输出目录不存在时创建目录

random.seed(21)  # 固定 Python 随机数种子，保证本次运行可复现
torch.manual_seed(21)  # 固定 PyTorch 随机数种子，保证随机增强可复现

source_path = os.path.join(INPUT_DIR, "sample.png")  # 函数⑥：拼接示例图片路径
if not os.path.exists(source_path):  # 函数⑦：判断示例图片是否已经存在
    image = Image.new("RGB", (320, 240), (40, 120, 200))  # 函数②：创建一张 RGB 示例图片
    draw = ImageDraw.Draw(image)  # 函数③：创建绘图对象
    draw.rectangle((40, 50, 160, 190), fill=(240, 180, 60))  # 绘制矩形，帮助观察翻转和裁剪
    draw.ellipse((190, 60, 290, 160), fill=(70, 210, 100))  # 绘制圆形，帮助观察增强结果
    image.save(source_path)  # 函数④：把示例图片保存到输入目录

non_image_path = os.path.join(INPUT_DIR, "说明.txt")  # 函数⑧：拼接非图片文件路径
non_image_file = open(non_image_path, "w", encoding="utf-8")  # 函数⑨：以写入模式打开非图片文件
non_image_file.write("这个文件用于测试过滤逻辑。")  # 函数⑩：写入非图片文件内容
non_image_file.close()  # 函数⑪：关闭文件，确保内容落盘

image_suffixes = {".jpg", ".jpeg", ".png", ".bmp"}  # 设置允许处理的图片后缀集合
input_files = []  # 创建空列表，用于保存筛选后的图片路径
for file_name in sorted(os.listdir(INPUT_DIR)):  # 函数⑫：遍历输入目录中的所有名称
    file_path = os.path.join(INPUT_DIR, file_name)  # 函数⑬：拼接当前文件的完整路径
    if os.path.isfile(file_path) and os.path.splitext(file_name)[1].lower() in image_suffixes:  # 函数⑭：判断是否为图片文件
        input_files.append(file_path)  # 把符合条件的图片路径加入列表

print("图片文件数量：", len(input_files))  # 打印过滤后的图片数量

resize = transforms.Resize((160, 160))  # 函数⑦：创建固定尺寸缩放变换
horizontal_flip = transforms.RandomHorizontalFlip(p=1.0)  # 函数⑧：创建必定水平翻转的变换
vertical_flip = transforms.RandomVerticalFlip(p=1.0)  # 函数⑨：创建必定垂直翻转的变换
rotation = transforms.RandomRotation(degrees=20, fill=(0, 0, 0))  # 函数⑩：创建随机旋转变换
brightness = transforms.ColorJitter(brightness=0.4)  # 函数⑪：创建亮度随机变化变换
crop = transforms.RandomResizedCrop(size=(160, 160), scale=(0.8, 1.0))  # 函数⑫：创建随机裁剪后缩放变换
blur = transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))  # 函数⑬：创建高斯模糊变换

transform = transforms.Compose([resize, horizontal_flip, rotation, brightness, crop, blur])  # 函数⑭：按顺序组合多个变换

for file_path in input_files:  # 遍历筛选后的图片文件
    image = Image.open(file_path).convert("RGB")  # 函数⑮：读取图片并统一转换为 RGB
    transformed_image = transform(image)  # 函数⑯：执行组合后的图像增强
    output_name = os.path.splitext(os.path.basename(file_path))[0] + "_augmented.png"  # 函数⑮：生成输出文件名
    output_path = os.path.join(OUTPUT_DIR, output_name)  # 函数⑯：拼接增强结果的输出路径
    transformed_image.save(output_path)  # 函数⑰：保存增强后的图片
    print("已保存：", output_name)  # 打印已保存的文件名

if len(input_files) > 0:  # 判断是否至少处理了一张图片
    print("输出尺寸：", transformed_image.size)  # 打印最后一张增强图片的尺寸
    print("输出模式：", transformed_image.mode)  # 打印最后一张增强图片的颜色模式
else:  # 没有图片时给出提示
    print("没有找到可处理的图片。")  # 打印空目录提示

print("非图片文件已跳过：", os.path.basename(non_image_path))  # 说明非图片文件没有进入处理列表
