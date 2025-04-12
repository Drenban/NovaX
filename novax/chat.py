import torch
from .model.model import MiniMind
from .model.tokenizer.tokenizer import MinimindTokenizer  # 根据实际路径调整
from .model.LMConfig import LMConfig

class NovaXChat:
    def __init__(self, model_path, config=None):
        """初始化 NovaX 聊天模型"""
        self.tokenizer = MinimindTokenizer()  # 使用默认配置
        self.config = config if config else LMConfig()  # 默认使用 LMConfig
        self.model = MiniMind(self.config)
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()

    def chat(self, text, max_length=50, device="cpu"):
        """生成对话回复"""
        self.model.to(device)
        tokens = self.tokenizer.encode(text)
        input_ids = torch.tensor([tokens], device=device)
        with torch.no_grad():
            output_ids = self.model.generate(input_ids, max_length=max_length)
        return self.tokenizer.decode(output_ids[0].tolist())

if __name__ == "__main__":
    # 测试代码
    bot = NovaXChat("path/to/model.pth")
    response = bot.chat("你好，我是 NovaX 用户")
    print(response)
