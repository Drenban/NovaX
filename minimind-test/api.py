from flask import Flask, request, jsonify
from minimind_chat import get_minimind_chat

app = Flask(__name__)
minimind = get_minimind_chat("path/to/minimind.pth")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    query = data.get("query", "")
    response = minimind.generate(query)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
