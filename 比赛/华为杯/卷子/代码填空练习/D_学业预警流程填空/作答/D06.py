# D06：只替换带题号的空位。
# 通过脚本位置找文件，PyCharm 的工作目录不同也能运行。
import os  # 处理路径与输出目录。
HERE = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在目录。
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))  # 随卷数据目录。
OUT = os.path.join(HERE, "outputs", "D06")  # 本题产物目录；不覆盖输入。
os.makedirs(OUT, exist_ok=True)  # 第一次运行也能保存。

import json
import pandas as pd
from llama_cpp import Llama

with open(os.path.join(DATA, 'risk_profiles.json'), encoding='utf-8') as file:
    profiles = json.load(file)
# 模型已移到“大模型调用”；优先查新位置，同时兼容旧目录。
project = os.path.normpath(os.path.join(DATA, '..', '..', '..', '..'))
model_name = 'qwen2-0_5b-instruct-q4_k_m.gguf'
candidates = [os.path.join(project, '大模型调用', '轻量量化模型', model_name),
              os.path.join(project, '卷子', '比赛相关', '轻量量化模型', model_name),
              os.path.join(project, '比赛相关', '轻量量化模型', model_name)]
model_path = os.environ.get('HUAWEI_GGUF')
if not model_path:
    for candidate in candidates:
        if os.path.isfile(candidate):
            model_path = candidate
            break
if not model_path or not os.path.isfile(model_path):
    checked = model_path or '；'.join(candidates)
    raise FileNotFoundError('未找到 GGUF，请设置 HUAWEI_GGUF。已检查：' + checked)
llm = Llama(model_path=__D06_01__, n_ctx=1536, n_threads=4, n_gpu_layers=0, seed=93, verbose=False)
facts_names = ['id', 'risk_probability', 'absences', 'study_hours']
allowed = ['联系学生核实缺勤', '安排学习计划沟通']

def unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('JSON 含重复字段')
        result[key] = value
    return result

def inspect_reply(raw, facts):
    try:
        parsed = json.__D06_02__(raw, object_pairs_hook=unique_keys)
    except (ValueError, TypeError):
        return None, 'invalid_json'
    expected = set(facts_names + ['action'])
    if not isinstance(parsed, dict) or __D06_03__:
        return None, 'wrong_fields'
    for field in facts_names:
        if __D06_04__:
            return None, 'wrong_facts'
    if __D06_05__:
        return None, 'wrong_action'
    return parsed, 'ok'

raw_responses = []
reports = []
checks = []
for facts in profiles:
    fact_text = json.dumps(__D06_06__)
    prompt = ('只输出一个 JSON 对象，不含代码围栏。复制以下全部字段和值：' + fact_text
              + '。再加 action 字段，只能是“联系学生核实缺勤”或“安排学习计划沟通”。')
    messages = [{'role': 'system', 'content': '你是学业支持助手，只用给定资料。'},
                {'role': 'user', 'content': prompt}]
    response = llm.__D06_07__(messages=messages, max_tokens=135, temperature=0.0)
    choice = response['choices'][0]
    raw = choice['message']['content'] or ''
    raw_responses.append({'id': facts['id'], 'prompt': prompt, 'raw': raw, 'finish_reason': choice.get('finish_reason')})
    parsed, issue = inspect_reply(raw, facts)
    if issue != 'ok':
        final = facts.copy()
        final['action'] = allowed[0]
        source = 'fallback'
    else:
        final = parsed.copy()
        source = 'model'
    final['source'] = source
    reports.append(final)
    checks.append({'id': facts['id'], 'check': issue, 'source': source})
with open(os.path.join(OUT, 'raw_responses.json'), 'w', encoding='utf-8') as file:
    json.dump(raw_responses, file, ensure_ascii=False, indent=2)
with open(os.path.join(OUT, 'reports.json'), 'w', encoding='utf-8') as file:
    json.dump(reports, file, ensure_ascii=False, indent=2)
pd.DataFrame(checks).to_csv(os.path.join(OUT, 'quality.csv'), index=False)
for row in checks:
    print('报告检查', row['id'], row['check'], '来源', row['source'])
