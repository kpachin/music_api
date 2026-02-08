from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib

app = Flask(__name__)
CORS(app)

# Load trained ML model
model = joblib.load("music_recommender.pkl")

# 🎵 Genre → Songs WITH YouTube Video IDs
GENRE_TO_SONGS = {

    "Reggae": [
        {"title": "Bob Marley – Three Little Birds", "video_id": "zaGUr6wzyT8"},
        {"title": "Chronixx – Skankin' Sweet", "video_id": "FjQ7JxJ5q8k"},
        {"title": "Protoje – Who Knows", "video_id": "KUbis_Zq8sM"}
    ],

    "Afrobeat": [
        {"title": "Burna Boy – Last Last", "video_id": "ZCxz-XwGZ5A"},
        {"title": "Wizkid – Essence", "video_id": "t_JnJdB8Hew"},
        {"title": "Davido – If", "video_id": "bJ7pMsoDn9o"}
    ],

    "Hip-hop": [
        {"title": "Drake – God's Plan", "video_id": "xpVfcZ0ZcFM"},
        {"title": "Kendrick Lamar – HUMBLE", "video_id": "tvTRZJ-4EyI"},
        {"title": "Sarkodie – Adonai", "video_id": "RyK0G8eZz8Y"}
    ],

    "Jazz": [
        {"title": "Miles Davis – So What", "video_id": "zqNTltOGh5c"},
        {"title": "John Coltrane – My Favorite Things", "video_id": "UlFNy9iWrpE"},
        {"title": "Herbie Hancock – Cantaloupe Island", "video_id": "8B1oIXGX0Io"}
    ],

    "Dancehall": [
        {"title": "Vybz Kartel – Fever", "video_id": "Z6kN2zB5y8c"},
        {"title": "Popcaan – Party Shot", "video_id": "0mP4h9uVJ_Y"},
        {"title": "Alkaline – Ocean Wave", "video_id": "w9t9fW6ZzJ8"}
    ],

    "Amapiano": [
        {"title": "Kabza De Small – Scorpion Kings", "video_id": "y8c9ZzKxG6Q"},
        {"title": "Focalistic – Ke Star", "video_id": "oN4xk0rY8R4"},
        {"title": "Asake – Terminator", "video_id": "s6A8v6lQmOQ"}
    ],

    "R&B": [
        {"title": "Chris Brown – Under The Influence", "video_id": "nM8j7nK1x9M"},
        {"title": "SZA – Snooze", "video_id": "Y8x9Zr6xG3I"},
        {"title": "Usher – Confessions", "video_id": "tIY3x8k5nP8"}
    ],

    "Highlife": [
        {"title": "E.T. Mensah – All For You", "video_id": "Z8c2B3x5JmY"},
        {"title": "Daddy Lumba – Theresa", "video_id": "s7Tn5YB1Z8M"},
        {"title": "Pat Thomas – Sika Ye Mogya", "video_id": "m2R8Z7G6kTQ"}
    ],

    "Pop": [
        {"title": "Taylor Swift – Shake It Off", "video_id": "nfWlot6h_JM"},
        {"title": "Ed Sheeran – Shape of You", "video_id": "JGwWNGJdvx8"},
        {"title": "Dua Lipa – Levitating", "video_id": "TUVcZfQe-Kw"}
    ],

    "Gospel": [
        {"title": "Joe Mettle – Bo Noo Ni", "video_id": "N7T5b7oR9Zk"},
        {"title": "Nathaniel Bassey – Imela", "video_id": "qv7c3Zp9G7M"},
        {"title": "Sinach – Way Maker", "video_id": "iJCV_2H9xD0"}
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

