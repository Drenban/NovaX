from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer
from novax.model.model import MiniMindLM
from novax.model.LMConfig import LMConfig
import os

app = Flask(__name__)

# MiniMind 单例
class MiniMindChat:
    def __init__(self, model_path, tokenizer_path):
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

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "minimind.pth")  # 模型存放路径
TOKENIZER_PATH = os.path.join(BASE_DIR, "novax", "model", "minimind_tokenizer")

minimind = MiniMindChat(MODEL_PATH, TOKENIZER_PATH)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    query = data.get("query", "")
    response = minimind.generate(query)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
