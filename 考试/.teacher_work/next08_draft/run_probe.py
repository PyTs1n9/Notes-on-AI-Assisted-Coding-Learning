import json
import os
import subprocess
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
STUDENT = 'D:/Conda_envs/envs/BaiQing3.10AI/python.exe'
TEACHER = 'D:/Code/Python/人工智能基础/.venv/Scripts/python.exe'
ENV = {key: value for key, value in os.environ.items() if key.upper() != 'PYTHONPATH'}
ENV.update(MPLBACKEND='Agg', PYTHONDONTWRITEBYTECODE='1', PYTHONIOENCODING='utf-8')
report = []
for label, executable in [('student', STUDENT), ('teacher', TEACHER)]:
    command = [executable, '-B', os.path.join(BASE, 'runtime_probe.py')]
    result = subprocess.run(command, cwd=BASE, env=ENV, capture_output=True, text=True, encoding='utf-8', timeout=60)
    with open(os.path.join(BASE, label + '_runtime_stdout.json'), 'w', encoding='utf-8') as file:
        file.write(result.stdout)
    with open(os.path.join(BASE, label + '_runtime_stderr.txt'), 'w', encoding='utf-8') as file:
        file.write(result.stderr)
    report.append({'command': command, 'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr})
print(json.dumps(report, ensure_ascii=False, indent=2))
