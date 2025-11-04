# ml_app/app.py
from flask import Flask, request, jsonify
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import os

app = Flask(__name__)

MODEL_PATH = 'iris_model.pkl'

def train_model():
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    print("Iris model trained and saved.")

# Train model on startup if not already present (for demo purposes)
if not os.path.exists(MODEL_PATH):
    train_model()
else:
    print("Iris model already exists.")

@app.route('/predict_iris', methods=['POST'])
def predict_iris():
    data = request.json
    sepal_length = data.get('sepal_length')
    sepal_width = data.get('sepal_width')
    petal_length = data.get('petal_length')
    petal_width = data.get('petal_width')

    if None in [sepal_length, sepal_width, petal_length, petal_width]:
        return jsonify({"error": "Missing one or more input features."}), 400

    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

    prediction = model.predict(features)[0]
    iris_names = load_iris().target_names
    predicted_species = iris_names[prediction]

    return jsonify({"predicted_species": predicted_species})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
