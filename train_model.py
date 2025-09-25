import pandas as pd
import numpy as np
import cv2
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib
import os

df = pd.read_csv('sample_cat_dog_dataset.csv')

data = []
labels = []

for _, row in df.iterrows():
    path = f'dataset/train/{row["filename"].split(".")[0]}/{row["filename"]}'
    if not os.path.exists(path):
        continue
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    features = hog(img, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), block_norm='L2-Hys')
    data.append(features)
    labels.append(row['label'])

X_train, X_test, y_train, y_test = train_test_split(np.array(data), np.array(labels), test_size=0.2)

model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/svm_model.pkl')
print("✅ Model trained and saved.")
