import pandas as pd
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load model
model = joblib.load("model.joblib")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    input_data = pd.DataFrame([data])
    prediction = model.predict(input_data)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
