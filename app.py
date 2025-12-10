from flask import Flask, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)

# Load your trained model
model = joblib.load("model.pkl")

@app.route("/predict", methods=["POST"])
def predict_genre():
    try:
        data = request.get_json()
        features = np.array(data["features"]).reshape(1, -1)
        prediction = model.predict(features)
        return jsonify({"recommended_genre": str(prediction[0])})

    except Exception as e:
        # This helps you see errors on Render logs
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def home():
    return "Music recommendation API is running!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
