from llama_cpp import Llama

llm = Llama(
	model_path="./qwen2-0_5b-instruct-q4_k_m.gguf",
	n_ctx=4096,  # 上下文窗口，0.5B推荐2048，不要直接拉32k
	n_gpu_layers=-1,  # -1=全部层丢GPU；CPU跑改成0；这个模型一共24层
	verbose=False  # 关掉底层日志
)

class Agent:
	def __init__(self, model):
		self.model = model
		self.messages = [{
			"role": "system",
			"content": """
        你是离线数据竞赛助手，擅长 Python、pandas 和 scikit-learn。
        回答规则：
        1. 优先给出可直接运行的 Python 代码。
        2. 代码必须使用 pandas、numpy、sklearn 等常见库。
        3. 不要虚构数据列名；缺少信息时先列出需要确认的字段。
        4. 每次只解决一个小问题。
        5. 对数据清洗操作说明可能造成的数据损失。
        6. 代码尽量短，不使用复杂语法。
        7. 输出格式固定为：
           【思路】
           【代码】
           【检查方法】
        """
		}]

	def run(self, user_input):
		result = self.model.create_chat_completion(
			messages=[{
				"role": "user", "content": user_input
			}],
			max_tokens=512
		)
		answer = result["choices"][0]["message"]["content"].strip()  # 提取并清理模型回答
		return answer  # 返回模型回答

agent = Agent(llm)
print("Qwen 终端助手（最近 10 轮上下文），输入 /exit 退出，/clear 清空上下文。")

while True:
	prompt = input("\n你: ").strip()
	if prompt:
		print("\n助手:", agent.run(prompt))
