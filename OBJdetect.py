import os
import base64
import uuid
from flask import Flask, request, jsonify
import openai
import time

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
app = Flask(__name__)
app.secret_key = os.urandom(24)

# Store object data by token (not by IP!)
last_objects = {}  # token: {"name": ..., "image_bytes": ...}

def get_object_description(img_bytes):
    img_b64 = base64.b64encode(img_bytes).decode()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text",
                     "text": "Describe only the main object in this image. Keep your answer very short and efficient. If it has a brand, product name, or visible text, include that in your answer. Reply with just the most specific identification possible, and nothing else."},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{img_b64}"}}
                ]
            }
        ],
        max_tokens=20
    )
    return response.choices[0].message.content.strip()

def ask_about_object(object_name, img_bytes, question):
    img_b64 = base64.b64encode(img_bytes).decode()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    f"You are a helpful robot assistant for product Q&A. The user has shown you this object: {object_name}. "
                    "ASSUME that any question you receive is about the detected object, unless it is very clearly about a completely unrelated topic or person. "
                    "Answer as briefly and directly as possible. "
                    "NEVER provide long explanations or lists. "
                    "ALWAYS answer in one or two short sentences maximum. "
                    "Answer as broadly and helpfully as possible about the object: its color, shape, type, use, brand, product family, price or where it can be bought, any visible text, similar products, and general facts. "
                    "If the question is open-ended (like 'tell me more', 'explain', 'what is this', etc.), reply with a brief summary or just the key features. "
                    "If you can guess the type (e.g., a Coke bottle, a smartphone), you may use your general knowledge, but also use what is visible in the image. "
                    "If the user asks for its price, give an estimate based on current market (e.g., on Amazon or a popular store). "
                    "If you really cannot answer because it's about something else (like 'who is Elon Musk'), reply: "
                    f"Sorry, I can only answer questions about the detected object: {object_name}. Please ask about it."
                )
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                ]
            }
        ],
        max_tokens=80
    )
    return response.choices[0].message.content.strip()

@app.route('/detect', methods=['POST'])
def detect():
    start = time.time()
    file = request.files.get('image')
    if file is None:
        return jsonify({'detected': False}), 400

    img_bytes = file.read()
    print(f"[Image Size] {len(img_bytes)} bytes ({len(img_bytes) / 1024:.2f} KB)")

    try:
        object_name = get_object_description(img_bytes)
        print("[OpenAI Vision Result]:", object_name)

        if object_name and len(object_name.strip()) > 0:
            token = str(uuid.uuid4())
            last_objects[token] = {"name": object_name, "image_bytes": img_bytes}
            desc = object_name
            return jsonify({"detected": True, "object": object_name, "desc": desc, "token": token})
        else:
            return jsonify({"detected": False})
    except Exception as e:
        print("[OpenAI Error]:", str(e))
        return jsonify({'detected': False, "error": str(e)}), 500

@app.route('/object_qa', methods=['POST'])
def object_qa():
    data = request.get_json()
    question = data.get("question", "").strip()
    token = data.get("token", None)
    info = last_objects.get(token)
    if not info:
        return jsonify({"answer": "I don't know what object you are referring to. Please show me the object first."})

    try:
        answer = ask_about_object(info["name"], info["image_bytes"], question)
        info["last_q"] = question
        info["last_a"] = answer
        last_objects[token] = info
        return jsonify({"answer": answer, "object": info["name"]})
    except Exception as e:
        print("[OpenAI QA Error]:", str(e))
        return jsonify({"answer": "Sorry, I couldn't answer that right now.", "error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
