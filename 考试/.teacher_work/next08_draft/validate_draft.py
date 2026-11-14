import hashlib
import json
import os
import re
import subprocess
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
DOC = os.path.join(BASE, '08_第八课：PyTorch基础一：Tensor与设备_草案.md')
STUDENT = 'D:/Conda_envs/envs/BaiQing3.10AI/python.exe'
TEACHER = 'D:/Code/Python/人工智能基础/.venv/Scripts/python.exe'
SOURCE_TOOL = 'C:/Users/Administrator/AppData/Local/hermes/skills/research/grounded-citations/scripts/sources.py'
GATE = 'C:/Users/Administrator/AppData/Local/hermes/skills/productivity/personal-tutor/scripts/verify_lesson_doc.py'
ENV = {key: value for key, value in os.environ.items() if key.upper() != 'PYTHONPATH'}
ENV.update(MPLBACKEND='Agg', PYTHONDONTWRITEBYTECODE='1', PYTHONIOENCODING='utf-8', TMPDIR=BASE, TEMP=BASE, TMP=BASE)
report = {'scope': BASE, 'status': 'draft_only_not_opened', 'environment_changed': False,
          'teacher_limitation': 'torch missing; no installation permitted',
          'child_environment': {'PYTHONPATH': 'removed_case_insensitively', 'MPLBACKEND': 'Agg',
                                'PYTHONDONTWRITEBYTECODE': '1', 'temporary_files': BASE},
          'runs': []}

def write(name, content):
    path = os.path.abspath(os.path.join(BASE, name))
    assert os.path.commonpath([BASE, path]) == BASE
    with open(path, 'w', encoding='utf-8', newline='\n') as file:
        file.write(content)

def run(label, args, input_text=None, env_updates=None):
    child_env = dict(ENV)
    if env_updates:
        child_env.update(env_updates)
    proc = subprocess.run(args, input=input_text, cwd=BASE, env=child_env, capture_output=True, text=True, encoding='utf-8', timeout=90)
    item = {'label': label, 'command': args, 'returncode': proc.returncode,
            'stdout_file': label + '_stdout.txt', 'stderr_file': label + '_stderr.txt'}
    write(item['stdout_file'], proc.stdout)
    write(item['stderr_file'], proc.stderr)
    report['runs'].append(item)
    print(label, 'exit', proc.returncode, flush=True)
    return proc

with open(DOC, encoding='utf-8') as file:
    text = file.read()
blocks = re.findall(r'^```python\n(.*?)^```\s*$', text, re.M | re.S)
assert len(blocks) == 1
code = blocks[0]
assert code.splitlines()[-1].startswith("print('原设备/目标设备/往返相等：'")
assert all('#' in line for line in code.splitlines() if line.strip())
write('demo.py', code)
execution = 'import sys; exec(compile(sys.stdin.read(), "document_first_fence", "exec"))'
student_demo = run('student_doc_demo', [STUDENT, '-B', '-c', execution], code)
assert student_demo.returncode == 0 and student_demo.stdout
cpu_demo = run('student_doc_demo_cuda_hidden', [STUDENT, '-B', '-c', execution], code, {'CUDA_VISIBLE_DEVICES': '-1'})
assert cpu_demo.returncode == 0 and cpu_demo.stdout
assert cpu_demo.stdout.splitlines()[:-1] == student_demo.stdout.splitlines()[:-1]
assert cpu_demo.stdout.splitlines()[-1] == '原设备/目标设备/往返相等： cpu cpu True'
teacher_demo = run('teacher_doc_demo', [TEACHER, '-B', '-c', execution], code)
assert teacher_demo.returncode != 0 and "No module named 'torch'" in teacher_demo.stderr
student_ref = run('student_reference_cpu', [STUDENT, '-B', os.path.join(BASE, 'private_reference.py')])
assert student_ref.returncode == 0 and student_ref.stdout
reference = json.loads(student_ref.stdout)
write('private_reference_output.json', json.dumps(reference, ensure_ascii=False, indent=2) + '\n')
auto_ref = run('student_reference_auto', [STUDENT, '-B', os.path.join(BASE, 'private_reference.py'), '--auto'])
assert auto_ref.returncode == 0
ref_auto = json.loads(auto_ref.stdout)
assert ref_auto['full_output_cpu'] == reference['full_output_cpu']
write('private_reference_auto_output.json', json.dumps(ref_auto, ensure_ascii=False, indent=2) + '\n')
teacher_ref = run('teacher_reference', [TEACHER, '-B', os.path.join(BASE, 'private_reference.py')])
assert teacher_ref.returncode != 0 and "No module named 'torch'" in teacher_ref.stderr
render = run('citations_render', [sys.executable, '-B', SOURCE_TOOL, '--ledger', os.path.join(BASE, 'citations.json'), 'render', '--style', 'plain'])
assert render.returncode == 0
acceptance = '''- 原输入shape为 `{input_shape}`，dtype为 `torch.uint8`；输入内容保持原样。
- 正确结果shape为 `{output_shape}`，dtype为 `{output_dtype}`；device为CPU或当前检测可用的CUDA设备，搬回CPU后元素保持一致。
- 完整数据比较必须支持判断，不只给shape；预测允许出错，但实跑后须按证据修正，不能把原预测当作结论。
- 类型顺序实验使用相同输入和偏移，区分“先计算再改浮点”与“先改浮点再计算”；最终正确结果保留应有的负值。
- 轴对齐实验在可表示负数的同一类型中进行；至少一次整体比较与一个像素的颜色对应证据共同支持解释。参数不能只因程序可运行就被认定为逐通道。
- 所有像素都满足原行列坐标对应、各颜色分量减去同色偏移的要求；不截断、不取绝对值、不把RGB倒置、不引入归一化。打印必要证据即可，无固定措辞或print数量要求。
- 本题未提供私有参考代码或正确像素答案；验收看真实计算与解释，不要求复刻演示的实现顺序。
'''.format(input_shape=tuple(reference['input_shape']), output_shape=tuple(reference['output_shape']), output_dtype=reference['output_dtype']).rstrip()
if '@@DEMO_OUTPUT@@' in text:
    text = text.replace('@@DEMO_OUTPUT@@', student_demo.stdout.rstrip())
    text = text.replace('@@ACCEPTANCE@@', acceptance)
    text = text.replace('@@SOURCES@@', render.stdout.strip())
    assert '@@' not in text
    write(os.path.basename(DOC), text)
else:
    assert student_demo.stdout.rstrip() in text
fresh = re.findall(r'^```python\n(.*?)^```\s*$', text, re.M | re.S)[0]
assert fresh == code
captured = re.findall(r'^```text\n(.*?)^```\s*$', text, re.M | re.S)[0]
assert captured.rstrip() == student_demo.stdout.rstrip()
for label, executable in [('student', STUDENT), ('teacher', TEACHER)]:
    gate = run(label + '_verify_lesson_doc', [sys.executable, '-B', GATE, DOC, '--python', executable])
    assert gate.returncode == (0 if label == 'student' else 1)
verify_sources = run('citations_verify', [sys.executable, '-B', SOURCE_TOOL, '--ledger', os.path.join(BASE, 'citations.json'), 'verify', DOC])
assert verify_sources.returncode == 0
exercise = text.split('## 【课后练习】')[1].split('#### 验收点')[0]
assert '```' not in exercise
assert not re.search(r'(torch\.[A-Za-z_]+|\.to|reshape|permute|unsqueeze)\s*\(', exercise)
cards = text.split('## 【核心内置函数拆解】')[1].split('### 教师核对来源')[0]
fields = ['作用', '完整语法', '关键参数', '返回值', '常见坑', '适用场景']
counts = {field: cards.count('**' + field + '**') for field in fields}
assert set(counts.values()) == {10}
assert text.count('## 【') == 6
assert not re.search(r'@@\w+@@', text)
report['checks'] = {'six_sections': True, 'one_python_fence': True,
    'all_demo_lines_commented': True, 'code_matches_demo_py': True, 'output_injected_from_exact_doc_code': True,
    'no_exercise_call_syntax': True, 'sizes_only_in_acceptance': True,
    'exercise_changed_decisions': ['uint8 arithmetic versus cast order', 'broadcast alignment after axis permutation'],
    'cards': counts, 'exercise_cpu_assertions': reference['assertions_passed'],
    'exercise_auto_assertions': ref_auto['assertions_passed'], 'cpu_auto_values_equal': True,
    'student_doc_gate': 'passed', 'teacher_doc_gate': 'blocked_missing_torch',
    'dual_environment_compatibility': 'not_established', 'cuda_unavailable_branch': 'passed_exact_doc_code_in_child_with_CUDA_VISIBLE_DEVICES_minus_1_no_global_change',
    'formal_files_git_archive_and_environments': 'untouched_by_this_task'}
report['artifact_sha256'] = {os.path.basename(DOC): hashlib.sha256(text.encode()).hexdigest(), 'demo.py': hashlib.sha256(code.encode()).hexdigest()}
write('teacher_validation.json', json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(report['checks'], ensure_ascii=False, indent=2))
