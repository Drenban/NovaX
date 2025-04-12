import torch
from transformers import AutoTokenizer
from model.model import MiniMindLM
from model.LMConfig import LMConfig

class MiniMindChat:
    def __init__(self, model_path, tokenizer_path="../NovaX/novax/model/minimind_tokenizer"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.config = LMConfig()
        self.model = MiniMindLM(self.config)
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()

    def generate(self, text, max_length=50, temperature=0.75, top_p=0.9):
        messages = [
            {"role": "system", "content": "你是 MiniMind，是一个有用的人工智能助手。"},
            {"role": "user", "content": text}
        ]
        formatted_input = self.tokenizer.apply_chat_template(messages, tokenize=False)
        inputs = self.tokenizer(formatted_input, return_tensors="pt")
        input_ids = inputs["input_ids"]

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
        response = self.tokenizer.decode(output_ids[0], skip_special_tokens=False)
        return response.split("<s>assistant\n")[-1].split("</s>")[0].strip()

# 单例模式，避免重复加载模型
_minimind_instance = None

def get_minimind_chat(model_path, tokenizer_path="../NovaX/novax/model/minimind_tokenizer"):
    global _minimind_instance
    if _minimind_instance is None:
        _minimind_instance = MiniMindChat(model_path, tokenizer_path)
    return _minimind_instance
