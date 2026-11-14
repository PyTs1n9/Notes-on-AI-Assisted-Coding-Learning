import inspect
import json
import sys

import torch

names = {
    'torch.tensor': torch.tensor,
    'torch.reshape': torch.reshape,
    'torch.permute': torch.permute,
    'torch.unsqueeze': torch.unsqueeze,
    'torch.equal': torch.equal,
    'torch.device': torch.device,
    'torch.Tensor.to': torch.Tensor.to,
    'torch.Tensor.__getitem__': torch.Tensor.__getitem__,
    'torch.cuda.is_available': torch.cuda.is_available,
}
result = {'python': sys.executable, 'python_version': sys.version, 'torch_version': torch.__version__,
          'cuda_available': torch.cuda.is_available(), 'apis': {}}
for name, function in names.items():
    item = {'text_signature': getattr(function, '__text_signature__', None)}
    try:
        item['inspect_signature'] = str(inspect.signature(function))
    except (ValueError, TypeError) as error:
        item['inspect_signature_unavailable'] = str(error)
    doc = inspect.getdoc(function) or ''
    item['docstring_head'] = '\n'.join(doc.splitlines()[:22])
    result['apis'][name] = item
raw = torch.tensor([[[7, 8, 9], [10, 11, 12]]], dtype=torch.uint8, device='cpu')
floating = raw.to(dtype=torch.float32)
chw = torch.permute(raw, (2, 0, 1))
batch = torch.unsqueeze(chw, 0)
result['behavior_checks'] = {
    'tensor_does_not_normalize': torch.equal(raw, floating),
    'dtype_checked_separately': floating.dtype == torch.float32 and raw.dtype == torch.uint8,
    'permute_coordinate_identity': torch.equal(raw[0, 1, :], chw[:, 0, 1]),
    'unsqueeze_preserves_data': torch.equal(batch[0], chw),
    'reshape_is_not_permute': not torch.equal(torch.reshape(raw, (3, 1, 2)), chw),
    'to_same_device_dtype_returns_self': raw.to(device='cpu', dtype=torch.uint8) is raw,
    'scalar_integer_index_shape': list(raw[0, 0, 0].shape),
    'slice_keeps_axis_shape': list(raw[0:1].shape),
    'uint8_subtract_wraps': (torch.tensor([1], dtype=torch.uint8) - torch.tensor([2], dtype=torch.uint8)).item() == 255,
    'broadcast_right_alignment': list((floating - torch.tensor([1, 2, 3], dtype=torch.float32)).shape),
}
assert all(value for value in result['behavior_checks'].values() if isinstance(value, bool))
print(json.dumps(result, ensure_ascii=False, indent=2))
