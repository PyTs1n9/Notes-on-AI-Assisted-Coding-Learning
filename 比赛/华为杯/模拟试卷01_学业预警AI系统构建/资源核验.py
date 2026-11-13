# 本文件只核验资源，不清洗数据、不训练模型、不生成考试答案。
import os  # 使用标准库处理当前文件的真实路径。
import json  # 读取随卷的资源清单。
import hashlib  # 重新计算资源校验值。

base_dir = os.path.dirname(os.path.abspath(__file__))  # 从脚本所在目录定位资料。
manifest_path = os.path.join(base_dir, '资源清单.json')  # 清单与脚本同目录。
with open(manifest_path, 'r', encoding='utf-8') as handle:  # 读取清单，不修改源文件。
    manifest = json.load(handle)  # 将 JSON 转成普通字典。
failed = 0  # 记录缺失或字节不一致的资源数量。
for item in manifest['files']:  # 逐个检查本卷命名的资源。
    file_path = os.path.normpath(os.path.join(base_dir, item['path']))  # 兼容上级共享模型路径。
    if not os.path.isfile(file_path):  # 不存在时如实报告，不联网补下载。
        print('[失败] 文件不存在：', item['path'])  # 输出可定位的相对路径。
        failed += 1  # 记入失败总数。
        continue  # 继续检查其他资源。
    digest = hashlib.sha256()  # 为本资源单独创建校验对象。
    with open(file_path, 'rb') as handle:  # 使用二进制模式读取原始字节。
        while True:  # 分块读取，避免把模型整体载入内存。
            chunk = handle.read(1024 * 1024)  # 每次最多读取一 MiB。
            if not chunk:  # 读到文件末尾时结束。
                break  # 退出分块循环。
            digest.update(chunk)  # 将本块加入校验。
    matches = digest.hexdigest() == item['sha256'] and os.path.getsize(file_path) == item['bytes']  # 同时核对大小和摘要。
    if matches:  # 只有两项均相同时才报告通过。
        print('[通过]', item['path'])  # 这是字节完整性结论，不是答卷通过。
    else:  # 发生字节变化时保留文件并报告。
        print('[失败] 字节或校验值变化：', item['path'])  # 不自动覆盖用户资源。
        failed += 1  # 累加本资源失败。
print('资源总数', len(manifest['files']), '失败数', failed)  # 输出实测计数。
raise SystemExit(1 if failed else 0)  # 让运行结果可以被自动核验。
