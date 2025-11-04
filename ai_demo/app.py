# ai_app/app.py
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/ask', methods=['POST'])
def ask_ai():
    data = request.json
    question = data.get('question', '').lower()

    if "hello" in question or "hi" in question:
        response = "Hello there! How can I help you today?"
    elif "weather" in question:
        response = "I cannot tell you the current weather, but it's always sunny in my world!"
    elif "buy" in question and "milk" in question:
        response = "Adding milk to your shopping list. Is there anything else?"
    else:
        response = "I'm a simple AI. I can only respond to a few pre-programmed questions."

    return jsonify({"answer": response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
