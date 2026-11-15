import torch  # 导入张量和上下文工具。
from torch import nn  # 导入线性层与随机失活层。
torch.manual_seed(23)  # 辅助⑰：固定本次探针的随机序列，不在每次前向前重置。
inputs = torch.ones(16, 2, dtype=torch.float32)  # 函数⑯：用全一输入让失活结果容易观察。
linear = nn.Linear(2, 2)  # 第十课已用：创建带可训练参数的线性层。
linear.eval()  # 函数②：先设置评估模式，它不关闭梯度。
eval_output = linear(inputs)  # 参数需要梯度，所以此时仍会建立计算图。
print('eval_linear_requires_grad', eval_output.requires_grad)  # 辅助⑰：读取输出属性验证eval的边界。
linear.train()  # 函数①：恢复训练模式。
with torch.no_grad():  # 函数③：只关闭这段运算的梯度记录。
    no_grad_output = linear(inputs)  # 参数仍存在，但此次运算不记录反向图。
    print('no_grad_keeps_training', linear.training)  # 辅助⑰：观察上下文并未把模型切为eval。
    print('no_grad_output_requires_grad', no_grad_output.requires_grad)  # 辅助⑰：输出不需要反向梯度。
dropout = nn.Dropout(p=0.5)  # 函数⑮：创建训练时会随机失活的独立层。
dropout.train()  # 函数①：显式开启随机失活行为。
with torch.no_grad():  # 函数③：关闭梯度不会停止训练模式下的随机失活。
    train_output = dropout(inputs)  # 函数⑮：本次得到随机失活后的输出。
dropout.eval()  # 函数②：评估时失活层保持输入。
with torch.inference_mode():  # 函数④：独立观察明确用于推理的上下文。
    eval_dropout_output = dropout(inputs)  # 函数⑮：此时不再置零和缩放输入。
print('train_dropout_preserves_input', torch.allclose(train_output, inputs, rtol=0.0, atol=0.0))  # 函数⑬：整体比较，不只看首个元素。
print('eval_dropout_preserves_input', torch.allclose(eval_dropout_output, inputs, rtol=0.0, atol=0.0))  # 函数⑬：评估模式保持全部元素。
print('inference_output_requires_grad', eval_dropout_output.requires_grad)  # 辅助⑰：推理结果不需要反向梯度。
