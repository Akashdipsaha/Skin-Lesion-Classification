from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
from model import Model
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# Load the model
model_path = '/Users/KIIT/Documents/skin_class/ENB1_8Class30.h5'  # Update with your model path
model = Model(model_path)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({"error": "No selected file"}), 400

            # Read the image
            img = Image.open(file.stream)
            img_array = np.array(img)

            # Make prediction
            prediction = model.predict(img_array)

            # Extract only the disease class
            if isinstance(prediction, tuple):
                class_pred, _ = prediction
            else:
                class_pred = prediction

            return jsonify({"disease_class": class_pred}), 200

        return jsonify({"error": "No valid input"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
