from flask import Flask, request, render_template, redirect, url_for, jsonify
from utils.preprocess import *
from utils.classify import *
from utils.ocr import *
from utils.summarize import *

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    image_file = request.files['image']

    if image_file.filename == '':
        return jsonify({"error": "Empty image filename"}), 400
    
    if image_file:
        image_bytes = image_file.read()
        image = Image.open(BytesIO(image_bytes))
        image.save('static/uploaded_image.png') 

        # text = tesseract_ocr(image_bytes)
        text = paddle_ocr(image_bytes)

        # Preprocess the text and classify it
        classification_result = classify_news(text)

        # Summarize
        summarized_news = extractive_summarize(text)

        return render_template('result.html', text=text, classification_result=classification_result, summarized_news = summarized_news, image_url='static/uploaded_image.png')

if __name__ == '__main__':
    app.run(debug=True)