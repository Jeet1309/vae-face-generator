from flask import Flask, jsonify, render_template
from generate import generate_image_base64

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['GET'])
def generate_face():
    img_b64 = generate_image_base64()
    return jsonify({'image': img_b64})

if __name__ == "__main__":
    app.run(debug=True)
