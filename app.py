from flask import Flask

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_genre():
    data = request.get_json()
    features = np.array(data["features"]).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({"recommended_genre": str(prediction[0])})

if __name__ == "__main__":
    app.run(debug=True)

