"""教师核验：执行传入的第10课脚本并检查完整张量、独立对象及实际训练结果。"""
import contextlib
import hashlib
import io
import json
import os
import runpy
import sys
import torch

source = os.path.abspath(sys.argv[1])
capture = io.StringIO()
with contextlib.redirect_stdout(capture):
    values = runpy.run_path(source)
checks = {}
checks['input_dtype_shape'] = values['x'].dtype == torch.float32 and tuple(values['x'].shape) == (4, 2)
checks['target_dtype_shape'] = values['y'].dtype == torch.int64 and values['y'].tolist() == [0, 0, 1, 1]
checks['five_losses_each'] = len(values['normal_losses']) == len(values['candidate_losses']) == 5
checks['same_full_initial_logits'] = torch.equal(values['normal_start'], values['candidate_start'])
checks['full_logit_shapes'] = all(tuple(values[key].shape) == (4, 2) for key in ['normal_start', 'normal_end', 'candidate_start', 'candidate_end'])
checks['normal_logits_changed'] = not torch.equal(values['normal_start'], values['normal_end'])
checks['candidate_logits_unchanged'] = torch.equal(values['candidate_start'], values['candidate_end'])
checks['candidate_losses_constant'] = len(set(values['candidate_losses'])) == 1
checks['normal_loss_decreased_observed'] = values['normal_losses'][-1] < values['normal_losses'][0]
normal_ids = set(id(p) for p in values['model'].parameters())
candidate_ids = set(id(p) for p in values['candidate_model'].parameters())
checks['independent_models'] = normal_ids.isdisjoint(candidate_ids)
checks['normal_optimizer_owns_normal'] = normal_ids == set(id(p) for group in values['optimizer'].param_groups for p in group['params'])
checks['candidate_optimizer_owns_candidate'] = candidate_ids == set(id(p) for group in values['candidate_optimizer'].param_groups for p in group['params'])
torch.manual_seed(7)
initial_model = values['SmallClassifier']()
checks['candidate_parameters_unchanged'] = all(torch.equal(initial_model.state_dict()[key], value) for key, value in values['candidate_model'].state_dict().items())
checks['normal_parameters_changed'] = any(not torch.equal(initial_model.state_dict()[key], value) for key, value in values['model'].state_dict().items())
checks['candidate_gradients_cleared'] = all(p.grad is None for p in values['candidate_model'].parameters())
assert all(checks.values()), checks
print(json.dumps({'source_sha256': hashlib.sha256(open(source, 'rb').read()).hexdigest(), 'torch': torch.__version__, 'device': str(values['x'].device), 'checks': checks, 'stdout': capture.getvalue()}, ensure_ascii=False, indent=2))
