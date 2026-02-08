from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib

app = Flask(__name__)
CORS(app)

# Load trained ML model
model = joblib.load("music_recommender.pkl")

# 🎵 Genre → Songs with YouTube IDs
GENRE_TO_SONGS = {

    "Reggae": [
        {"title": "Bob Marley – Three Little Birds", "youtube_id": "LanCLS_hIo4"},
        {"title": "Chronixx – Skankin' Sweet", "youtube_id": "g0QYw4FhD2k"},
        {"title": "Protoje – Who Knows", "youtube_id": "C3k8hZsZ9nE"}
    ],

    "Afrobeat": [
        {"title": "Burna Boy – Last Last", "youtube_id": "421w1j87fEM"},
        {"title": "Wizkid – Essence", "youtube_id": "7a5Vt0U2L8E"},
        {"title": "Davido – If", "youtube_id": "iWQ5H8m8n5c"}
    ],

    "Hip-hop": [
        {"title": "Drake – God's Plan", "youtube_id": "xpVfcZ0ZcFM"},
        {"title": "Kendrick Lamar – HUMBLE.", "youtube_id": "tvTRZJ-4EyI"},
        {"title": "Sarkodie – Adonai", "youtube_id": "2lD8VwY1KFE"}
    ],

    "Jazz": [
        {"title": "Miles Davis – So What", "youtube_id": "zqNTltOGh5c"},
        {"title": "John Coltrane – My Favorite Things", "youtube_id": "qWG2dsXV5HI"},
        {"title": "Herbie Hancock – Cantaloupe Island", "youtube_id": "8B1oIXGX0Io"}
    ],

    "Dancehall": [
        {"title": "Vybz Kartel – Fever", "youtube_id": "P6IOVgC-IUg"},
        {"title": "Popcaan – Party Shot", "youtube_id": "QG1J9Z6a3YA"},
        {"title": "Alkaline – Ocean Wave", "youtube_id": "7q4tQZL9uYc"}
    ],

    "Amapiano": [
        {"title": "Kabza De Small – Scorpion Kings", "youtube_id": "cFfE9j6L7W4"},
        {"title": "Focalistic – Ke Star", "youtube_id": "8ZKZL9u0h8M"},
        {"title": "Asake – Terminator", "youtube_id": "a0xJ0Qp0p1Q"}
    ],

    "R&B": [
        {"title": "Chris Brown – Under The Influence", "youtube_id": "V3kJQ1JjF5Q"},
        {"title": "SZA – Snooze", "youtube_id": "L0CzK2X5kYo"},
        {"title": "Usher – Confessions", "youtube_id": "5Sy19X0xxrM"}
    ],

    "Highlife": [
        {"title": "E.T. Mensah – All For You", "youtube_id": "4gGkPzN1ZgE"},
        {"title": "Daddy Lumba – Theresa", "youtube_id": "Rj5T3n3j3P4"},
        {"title": "Pat Thomas – Sika Ye Mogya", "youtube_id": "k9LQz7mF2xM"}
    ],

    "Pop": [
        {"title": "Taylor Swift – Shake It Off", "youtube_id": "nfWlot6h_JM"},
        {"title": "Ed Sheeran – Shape of You", "youtube_id": "JGwWNGJdvx8"},
        {"title": "Dua Lipa – Levitating", "youtube_id": "TUVcZfQe-Kw"}
    ],

    "Gospel": [
        {"title": "Joe Mettle – Bo Noo Ni", "youtube_id": "k3F5KZy8z4I"},
        {"title": "Nathaniel Bassey – Imela", "youtube_id": "Rz9p6nXJz5E"},
        {"title": "Sinach – Way Maker", "youtube_id": "iJCV_2H9xD0"}
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
            [{"title": "No songs available", "youtube_id": ""}]
        )

        return jsonify({
            "recommended_genre": genre,
            "songs": songs
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def home():
    return "🎵 Music Recommendation API is running!"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
