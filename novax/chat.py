import torch
from transformers import AutoTokenizer
from .model.model import MiniMindLM
from .model.LMConfig import LMConfig

class NovaXChat:
    def __init__(self, model_path, tokenizer_path="novax/model/minimind_tokenizer", config=None):
        """初始化 NovaX 聊天模型"""
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        # 加载模型配置和权重
        self.config = config if config else LMConfig()
        self.model = MiniMindLM(self.config)
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()

    def chat(self, text, max_length=50, device="cpu", temperature=0.75, top_p=0.9):
        """生成对话回复"""
        self.model.to(device)
        # 使用 chat_template 格式化输入
        messages = [
            {"role": "system", "content": "你是 MiniMind，是一个有用的人工智能助手。"},
            {"role": "user", "content": text}
        ]
        formatted_input = self.tokenizer.apply_chat_template(messages, tokenize=False)
        # 分词
        inputs = self.tokenizer(formatted_input, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        # 生成输出
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        # 解码并返回
        full_output = self.tokenizer.decode(output_ids[0], skip_special_tokens=False)
        # 提取 assistant 部分
        response = full_output.split("<s>assistant\n")[-1].split("</s>")[0]
        return response.strip()

if __name__ == "__main__":
    # 测试代码
    bot = NovaXChat("path/to/model.pth")
    response = bot.chat("你好，我是 NovaX 用户")
    print(response)
