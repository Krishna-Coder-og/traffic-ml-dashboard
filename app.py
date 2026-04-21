from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained model and feature columns
model = joblib.load('model.pkl')
model_columns = joblib.load('model_columns.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        print("Received Input:", data)
        # Convert input to DataFrame
        df = pd.DataFrame([data])
        
        # Apply dummies (same as training)
        df = pd.get_dummies(df)
        
        # Re-align dummy columns with training columns
        # Fill missing columns with 0
        df = df.reindex(columns=model_columns, fill_value=0)
        
        # Predict
        prediction = model.predict(df)
        
        return jsonify({'prediction': int(prediction[0])})
        
    except Exception as e:
        print("Error:", e)
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(port=5000, debug=True)
