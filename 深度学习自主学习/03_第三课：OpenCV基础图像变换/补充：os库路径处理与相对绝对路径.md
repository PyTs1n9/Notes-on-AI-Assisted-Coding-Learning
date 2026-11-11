# 补充：`os` 库路径处理与相对路径、绝对路径

这份补充专门解决一个常见问题：代码里的路径看起来只是一个字符串，但它到底从哪里开始找、为什么换个运行位置就找不到文件、什么时候应该用相对路径，什么时候应该用绝对路径。

本文件的所有示例都使用 Windows 路径，并以本课程第三课的图片为实际案例。

## 一、先建立三个概念

### 1. 路径字符串

路径字符串只是文字，例如：

- `input\\sample.png`
- `D:\\Code\\Python\\人工智能基础\\深度学习自主学习\\03_第三课：OpenCV基础图像变换\\input\\sample.png`

它本身不会自动读取文件。只有当你把它交给 `open()`、`cv2.imread()`、`os.path.exists()` 等函数时，程序才会根据它寻找目标。

### 2. 当前工作目录 `current working directory`

当前工作目录简称 **cwd**。它是程序启动时所在的目录。

可以用下面的代码查看：

```python
import os  # 导入 os，用于查看当前工作目录

print(os.getcwd())  # 函数：获取当前工作目录
```

如果当前工作目录是：

```text
D:\Code\Python\人工智能基础\深度学习自主学习\03_第三课：OpenCV基础图像变换
```

那么：

```python
"input/sample.png"
```

实际指向的就是：

```text
D:\Code\Python\人工智能基础\深度学习自主学习\03_第三课：OpenCV基础图像变换\input\sample.png
```

如果当前工作目录换成：

```text
C:\Users\Administrator\AppData\Local\hermes\cache\scratch
```

同一个：

```python
"input/sample.png"
```

就会被解释成：

```text
C:\Users\Administrator\AppData\Local\hermes\cache\scratch\input\sample.png
```

所以相对路径能不能找到文件，首先取决于当前工作目录。

### 3. 脚本所在目录

脚本所在目录是 `.py` 文件实际保存的位置。它和当前工作目录可能相同，也可能不同。

例如：

- 脚本文件在课程目录中；
- 你却从 PyCharm 的另一个工作目录启动它。

这时：

```python
os.getcwd()
```

得到的是启动位置，而不是脚本位置。

如果想让代码从“脚本旁边的 input 文件夹”找文件，应该使用：

```python
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
```

这行的含义是：

1. `__file__`：当前 Python 文件的路径；
2. `os.path.abspath(__file__)`：把它转换成绝对路径；
3. `os.path.dirname(...)`：取出它所在的文件夹。

## 二、相对路径和绝对路径的区别

### 1. 相对路径：从当前参照位置出发

相对路径不从盘符开始。例如：

```python
"input/sample.png"
```

它只说明“当前参照目录下面的 `input` 文件夹里的 `sample.png`”。

相对路径的优点：

- 短；
- 代码可以换电脑、换课程目录继续使用；
- 适合课程项目、团队项目和需要打包分享的代码。

相对路径的风险：

- 换工作目录后可能找不到文件；
- PyCharm、命令行、IDE 运行配置可能使用不同的工作目录；
- 报错时经常看到 `FileNotFoundError` 或 OpenCV 读取失败。

### 2. 绝对路径：从盘符或根目录写起

绝对路径从盘符开始，例如：

```python
r"D:\Code\Python\人工智能基础\深度学习自主学习\03_第三课：OpenCV基础图像变换\input\sample.png"
```

前面的 `r` 表示原始字符串（raw string）。它可以减少 Windows 反斜杠转义带来的麻烦。

绝对路径的优点：

- 指向明确；
- 调试时容易确认程序到底找哪个文件；
- 适合一次性检查本机文件和定位问题。

绝对路径的风险：

- 换电脑后通常失效；
- 换用户名、盘符或课程目录后需要修改；
- 不适合直接写死在需要分享的项目中。

### 3. 实际选择规则

可以直接按下面的规则判断：

| 场景 | 推荐方式 | 原因 |
|---|---|---|
| 课程项目内部读取 `input/` | `__file__` + `os.path.join()` | 不依赖启动位置，项目可以移动 |
| 调试某个本机文件是否存在 | 临时使用绝对路径 | 最容易确认目标位置 |
| 项目需要发给别人 | 相对路径或基于 `__file__` 的路径 | 不绑定你的电脑目录 |
| 输出到当前项目的 `output/` | `BASE_DIR` + `os.path.join()` | 输入输出都跟随项目移动 |
| 只在当前目录执行的小脚本 | 简单相对路径 | 代码短，前提是工作目录固定 |
| 课程代码在 PyCharm 中运行 | `__file__` + `os.path.join()` | 避免 PyCharm 工作目录设置影响读取 |

最实用的结论是：**绝对路径适合定位问题，相对路径适合共享项目；项目代码中最好使用“脚本目录 + 相对位置”来构建稳定路径。**

## 三、`os.path.join()`：不要手动拼斜杠

### 1. 不推荐的写法

```python
file_path = "D:\\Code\\Python\\人工智能基础\\深度学习自主学习\\03_第三课：OpenCV基础图像变换\\input\\sample.png"
```

这段代码虽然可能运行，但路径很长，换目录时不容易修改。

下面这种写法更容易出错：

```python
file_path = base_dir + "\\input\\sample.png"
```

常见问题包括：

- `base_dir` 末尾已经有 `\\`，结果出现重复斜杠；
- 忘记写斜杠，目录名和文件名粘在一起；
- Windows 和其他系统的路径分隔符不同。

### 2. 推荐的写法

```python
file_path = os.path.join(base_dir, "input", "sample.png")
```

`os.path.join()` 会按照当前操作系统的规则拼接路径。

在本课中，推荐固定写成：

```python
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 找到当前脚本目录
INPUT_DIR = os.path.join(BASE_DIR, "input")  # 拼出 input 文件夹
OUTPUT_DIR = os.path.join(BASE_DIR, "output")  # 拼出 output 文件夹
INPUT_FILE = os.path.join(INPUT_DIR, "sample.png")  # 拼出图片文件
```

这四行解决了一个完整问题：无论从哪里启动脚本，都从脚本旁边寻找 `input/sample.png`。

## 四、几个最常用的路径函数区别

### 1. `os.getcwd()`：我从哪里启动程序

```python
current_dir = os.getcwd()
```

它返回当前工作目录。

它回答的问题是：

> 程序现在把哪个文件夹当作相对路径的起点？

它不一定是脚本所在目录。

### 2. `os.path.abspath(path)`：把路径转成绝对路径

```python
absolute_path = os.path.abspath("input/sample.png")
```

它会根据当前工作目录，把相对路径转换成绝对路径。

注意：它不会自动修正“错误的起点”。如果当前工作目录本来就不对，那么得到的绝对路径也可能指向错误位置。

例如：

```python
os.path.abspath("input/sample.png")
```

和：

```python
os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "input", "sample.png"))
```

虽然都用了 `abspath()`，但参照起点不同：

- 第一种以当前工作目录为起点；
- 第二种先找到脚本目录，再以脚本目录为起点。

课程代码通常应该使用第二种思路。

### 3. `os.path.dirname(path)`：取父目录

```python
parent_dir = os.path.dirname("D:\\project\\input\\sample.png")
```

结果是：

```text
D:\project\input
```

它回答的问题是：

> 这个文件或路径位于哪个文件夹？

### 4. `os.path.basename(path)`：取最后的名称

```python
file_name = os.path.basename("D:\\project\\input\\sample.png")
```

结果是：

```text
sample.png
```

它回答的问题是：

> 这个路径最后的文件名或文件夹名是什么？

### 5. `os.path.splitext(path)`：拆分主名和扩展名

```python
name, suffix = os.path.splitext("sample.png")
```

结果是：

```python
name == "sample"
suffix == ".png"
```

它适合做两类事情：

- 判断文件扩展名；
- 根据原文件名生成新文件名。

例如：

```python
base_name = os.path.splitext(os.path.basename(file_path))[0]
output_name = base_name + "_gray.png"
```

如果 `file_path` 是 `input/sample.png`，输出文件名就是：

```text
sample_gray.png
```

### 6. `os.path.exists(path)`、`isfile(path)`、`isdir(path)`

这三个函数都用于检查，但检查对象不同：

| 函数 | 检查内容 | 例子 |
|---|---|---|
| `os.path.exists(path)` | 路径是否存在 | 文件或文件夹都可以返回 `True` |
| `os.path.isfile(path)` | 是否是普通文件 | 文件返回 `True`，文件夹返回 `False` |
| `os.path.isdir(path)` | 是否是文件夹 | 文件夹返回 `True`，普通文件返回 `False` |

典型用法：

```python
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
```

更简洁的写法是：

```python
os.makedirs(output_dir, exist_ok=True)
```

`exist_ok=True` 表示：如果目录已经存在，也不要报错。

### 7. `os.path.normpath(path)`：整理路径格式

```python
messy_path = os.path.join(lesson_dir, "input", "..", "input", "sample.png")
clean_path = os.path.normpath(messy_path)
```

`..` 表示返回上一级目录。`normpath()` 会把这些路径符号整理成更规范的形式。

它主要用于：

- 清理重复的路径片段；
- 处理 `.` 和 `..`；
- 让路径显示得更清楚。

### 8. `os.path.relpath(path, start)`：计算相对路径

```python
relative_path = os.path.relpath(input_file, lesson_dir)
```

它回答的问题是：

> 从 `lesson_dir` 出发，怎样用相对路径走到 `input_file`？

例如：

```text
input\sample.png
```

它不会移动文件，只是计算路径写法。

## 五、最推荐的课程项目写法

### 案例：OpenCV 第三课读取输入、创建输出并保存结果

```python
import os  # 导入 os，用于处理路径和目录
import cv2  # 导入 OpenCV，用于读取和保存图片

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 函数：定位当前脚本所在目录
INPUT_DIR = os.path.join(BASE_DIR, "input")  # 函数：拼出输入目录
OUTPUT_DIR = os.path.join(BASE_DIR, "output")  # 函数：拼出输出目录
INPUT_FILE = os.path.join(INPUT_DIR, "sample.png")  # 函数：拼出输入图片路径
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "sample_copy.png")  # 函数：拼出输出图片路径

os.makedirs(OUTPUT_DIR, exist_ok=True)  # 函数：创建输出目录，已存在时不报错

if not os.path.isfile(INPUT_FILE):  # 函数：检查输入路径是否是普通文件
    raise FileNotFoundError("找不到输入图片：" + INPUT_FILE)  # 把实际检查的路径写进错误信息

image = cv2.imread(INPUT_FILE, cv2.IMREAD_COLOR)  # 使用绝对路径读取图片
if image is None:  # 检查 OpenCV 是否解码成功
    raise ValueError("图片读取失败：" + INPUT_FILE)  # 说明文件存在但读取可能失败

success = cv2.imwrite(OUTPUT_FILE, image)  # 使用绝对路径保存图片
print("输入路径：", INPUT_FILE)  # 打印实际读取的路径
print("输出路径：", OUTPUT_FILE)  # 打印实际保存的路径
print("保存成功：", success)  # 检查 OpenCV 的保存返回值
```

这个案例的关键不是“把路径写得很长”，而是：

- 用 `__file__` 找到脚本目录；
- 用 `os.path.join()` 拼接项目内部位置；
- 用 `os.makedirs()` 确保输出目录存在；
- 用 `isfile()` 检查输入文件；
- 用实际路径打印帮助定位问题。

## 六、实际运行案例

下面的程序在 `BaiQing3.10AI` 环境中运行，使用课程第三课的真实文件位置和图片文件。

```python
import os  # 导入 os，用于路径、目录和文件处理

ROOT_DIR = r"D:\Code\Python\人工智能基础"  # 设置课程根目录的绝对路径
LESSON_DIR = os.path.join(ROOT_DIR, "深度学习自主学习", "03_第三课：OpenCV基础图像变换")  # 拼出第三课目录
INPUT_FILE = os.path.join(LESSON_DIR, "input", "sample.png")  # 拼出图片的绝对路径
SCRATCH_DIR = r"C:\Users\Administrator\AppData\Local\hermes\cache\scratch\os_path_supplement"  # 设置临时演示目录

print("当前工作目录：", os.getcwd())  # 函数①：查看相对路径的起点
print("课程根目录：", ROOT_DIR)  # 打印课程根目录
print("课程根目录存在：", os.path.exists(ROOT_DIR))  # 函数②：检查路径是否存在
print("课程根目录是文件夹：", os.path.isdir(ROOT_DIR))  # 函数③：检查路径是否是文件夹
print("图片绝对路径：", INPUT_FILE)  # 打印拼出的完整路径
print("图片存在：", os.path.exists(INPUT_FILE))  # 函数②：检查图片路径是否存在
print("图片是文件：", os.path.isfile(INPUT_FILE))  # 函数③：检查图片是否是普通文件

relative_input = os.path.join("input", "sample.png")  # 函数④：生成一个相对路径
print("相对路径字符串：", relative_input)  # 打印相对路径本身
print("相对路径转绝对路径：", os.path.abspath(relative_input))  # 函数⑤：根据当前工作目录转换
print("脚本目录定位示例：", os.path.dirname(os.path.abspath(__file__)))  # 函数⑤⑥：定位当前脚本目录

print("文件名：", os.path.basename(INPUT_FILE))  # 函数⑦：取最后的文件名
print("父目录：", os.path.dirname(INPUT_FILE))  # 函数⑥：取文件所在目录
print("拆分文件名：", os.path.splitext(os.path.basename(INPUT_FILE)))  # 函数⑦⑧：拆分主名和扩展名
print("规范化路径：", os.path.normpath(os.path.join(LESSON_DIR, "input", "..", "input", "sample.png")))  # 函数④⑨：整理路径
print("相对课程目录的路径：", os.path.relpath(INPUT_FILE, LESSON_DIR))  # 函数⑩：计算相对路径

old_cwd = os.getcwd()  # 保存原来的工作目录
os.chdir(LESSON_DIR)  # 函数⑪：临时切换工作目录
print("切换后的工作目录：", os.getcwd())  # 函数①：查看切换结果
print("切换后相对路径存在：", os.path.exists(relative_input))  # 函数②：验证相对路径可用
os.chdir(old_cwd)  # 函数⑪：恢复原来的工作目录
print("恢复后的工作目录：", os.getcwd())  # 函数①：确认已经恢复

new_dir = os.path.join(SCRATCH_DIR, "demo_output", "nested")  # 函数④：拼出多层输出目录
os.makedirs(new_dir, exist_ok=True)  # 函数⑫：创建多层目录
print("创建目录后存在：", os.path.isdir(new_dir))  # 函数③：验证目录创建成功
print("拼接输出文件：", os.path.join(new_dir, "result.png"))  # 函数④：拼出输出文件路径
```

### 真实运行结果

运行环境：`D:\Conda_envs\envs\BaiQing3.10AI\python.exe`

```text
当前工作目录： C:\Users\Administrator\AppData\Local\hermes\cache\scratch\os_path_supplement
课程根目录： D:\Code\Python\人工智能基础
课程根目录存在： True
课程根目录是文件夹： True
图片绝对路径： D:\Code\Python\人工智能基础\深度学习自主学习\03_第三课：OpenCV基础图像变换\input\sample.png
图片存在： True
图片是文件： True
相对路径字符串： input\sample.png
相对路径转绝对路径： C:\Users\Administrator\AppData\Local\hermes\cache\os_path_supplement\input\sample.png
脚本目录定位示例： C:\Users\Administrator\AppData\Local\hermes\cache\os_path_supplement
文件名： sample.png
父目录： D:\Code\Python\人工智能基础\深度学习自主学习\03_第三课：OpenCV基础图像变换\input
拆分文件名： ('sample', '.png')
规范化路径： D:\Code\Python\人工智能基础\深度学习自主学习\03_第三课：OpenCV基础图像变换\input\sample.png
相对课程目录的路径： input\sample.png
切换后的工作目录： D:\Code\Python\人工智能基础\深度学习自主学习\03_第三课：OpenCV基础图像变换
切换后相对路径存在： True
恢复后的工作目录： C:\Users\Administrator\AppData\Local\hermes\cache\os_path_supplement
创建目录后存在： True
拼接输出文件： C:\Users\Administrator\AppData\Local\hermes\cache\os_path_supplement\demo_output\nested\result.png
```

> 说明：上面的临时目录名称是本次真实运行产生的目录名。你自己运行时，如果临时目录位置不同，相关输出会随实际环境变化；这不影响路径函数的使用方式。

## 七、三个最容易混淆的案例

### 案例一：为什么 `os.getcwd()` 不是脚本目录？

因为 `os.getcwd()` 看的是“程序从哪里启动”，不是“当前 `.py` 文件保存在哪里”。

```python
print(os.getcwd())
```

如果 PyCharm 的工作目录配置错了，它就会显示错误的相对路径起点。

要定位脚本旁边的文件，应使用：

```python
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
```

### 案例二：为什么 `os.path.abspath("input/sample.png")` 仍然可能错？

因为 `abspath()` 只是把当前相对路径展开，它仍然依赖当前工作目录。

错误起点：

```text
C:\\wrong_folder
```

调用：

```python
os.path.abspath("input/sample.png")
```

得到的就会是：

```text
C:\\wrong_folder\\input\\sample.png
```

正确思路是先确定参照目录：

```python
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.join(BASE_DIR, "input", "sample.png")
```

### 案例三：为什么目录存在，OpenCV 还是读取失败？

需要分开检查三个问题：

```python
print(os.path.exists(input_file))  # 路径是否存在
print(os.path.isfile(input_file))  # 是否是普通文件
image = cv2.imread(input_file)  # OpenCV 是否成功解码
print(image is None)  # True 表示解码失败
```

可能出现：

- 路径不存在；
- 路径指向的是文件夹；
- 文件存在，但文件损坏；
- 路径含特殊字符，当前库或环境读取失败；
- 图片格式或权限导致读取失败。

所以 `exists=True` 只能说明路径存在，不能单独证明图片已经成功读取。

## 八、最终记忆框架

遇到路径问题时，按这个顺序检查：

1. **我现在从哪里启动？**——查看 `os.getcwd()`。
2. **我要找的完整路径是什么？**——用 `os.path.abspath()` 或打印 `os.path.join()` 的结果。
3. **这个路径存在吗？**——查看 `os.path.exists()`。
4. **它是文件还是文件夹？**——分别查看 `os.path.isfile()` 和 `os.path.isdir()`。
5. **项目是否应该跟随脚本移动？**——如果应该，使用 `__file__` + `os.path.dirname()` + `os.path.join()`。
6. **输出目录是否提前创建？**——使用 `os.makedirs(..., exist_ok=True)`。

最后只记一句：

> **调试时可以先用绝对路径确认位置；写项目代码时，用脚本目录作为起点，再用 `os.path.join()` 拼出项目内部的相对位置。**
