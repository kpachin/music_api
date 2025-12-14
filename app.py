from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib

app = Flask(__name__)
CORS(app)

# Load trained ML model
model = joblib.load("model.pkl")

# 🎵 Genre → Real Songs Mapping
GENRE_TO_SONGS = {

    "Reggae": [
        "Bob Marley – Three Little Birds",
        "Chronixx – Skankin' Sweet",
        "Protoje – Who Knows"
    ],

    "Afrobeat": [
        "Burna Boy – Last Last",
        "Wizkid – Essence",
        "Davido – If"
    ],

    "Hip-hop": [
        "Drake – God's Plan",
        "Kendrick Lamar – HUMBLE",
        "Sarkodie – Adonai"
    ],

    "Jazz": [
        "Miles Davis – So What",
        "John Coltrane – My Favorite Things",
        "Herbie Hancock – Cantaloupe Island"
    ],

    "Dancehall": [
        "Vybz Kartel – Fever",
        "Popcaan – Party Shot",
        "Alkaline – Ocean Wave"
    ],

    "Amapiano": [
        "Kabza De Small – Scorpion Kings",
        "Focalistic – Ke Star",
        "Asake – Terminator"
    ],

    "R&B": [
        "Chris Brown – Under The Influence",
        "SZA – Snooze",
        "Usher – Confessions"
    ],

    "Highlife": [
        "E.T. Mensah – All For You",
        "Daddy Lumba – Theresa",
        "Pat Thomas – Sika Ye Mogya"
    ],

    "Pop": [
        "Taylor Swift – Shake It Off",
        "Ed Sheeran – Shape of You",
        "Dua Lipa – Levitating"
    ],

    "Gospel": [
        "Joe Mettle – Bo Noo Ni",
        "Nathaniel Bassey – Imela",
        "Sinach – Way Maker"
    ]
}

@app.route("/predict", methods=["POST"])
def predict_genre():
    try:
        data = request.get_json()

        features = np.array(data["features"]).reshape(1, -1)
        prediction = model.predict(features)

        genre = str(prediction[0])

        songs = GENRE_TO_SONGS.get(
            genre,
            ["No songs available for this genre"]
        )

        return jsonify({
            "recommended_genre": genre,
            "songs": songs
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def home():
    return "Music Recommendation API is running!"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)