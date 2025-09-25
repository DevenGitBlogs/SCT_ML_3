from flask import Flask, request, render_template, jsonify
import joblib, os, cv2
import numpy as np
from skimage.feature import hog

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
model = joblib.load('model/svm_model.pkl')

def extract_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    features = hog(img, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), block_norm='L2-Hys')
    return np.array([features])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['image']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    features = extract_features(filepath)
    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0].max()
    label = 'Dog' if pred == 1 else 'Cat'

    return jsonify({'result': label, 'confidence': f'{prob*100:.2f}%'})

if __name__ == '__main__':
    app.run(debug=True)
