import datetime
import os
import pickle
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tensorflow import keras
from pymongo import MongoClient
import numpy as np
import pandas as pd


def get_database():
    MONGODB_URI = os.environ.get('MONGODB_URI')
    if not MONGODB_URI:
        MONGODB_URI = "mongodb://root:password@localhost:27017/ium?authSource=admin"

    client = MongoClient(MONGODB_URI)

    return client['ium']


db = get_database()
predictions_collection = db['predictions']

app = FastAPI()

origins = ["*"]
methods = ["*"]
headers = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=methods,
    allow_headers=headers
)

# loading model 1
with open("./models/model.pickle", "rb") as f:
    model_1 = pickle.load(f)

with open("./models/model_mlb.pickle", "rb") as f:
    mlb_1 = pickle.load(f)

with open('./models/model_genre_to_cluster.pickle', 'rb') as f:
    genre_to_cluster_1 = pickle.load(f)

with open('./models/model_clustered_genres.pickle', 'rb') as f:
    clustered_genres_1 = pickle.load(f)

# loading model 2
model_2 = keras.models.load_model("./models/model2.h5")

with open("./models/model2_mlb.pickle", "rb") as f:
    mlb_2 = pickle.load(f)

with open('./models/model2_genre_to_cluster.pickle', 'rb') as f:
    genre_to_cluster_2 = pickle.load(f)

with open('./models/model2_clustered_genres.pickle', 'rb') as f:
    clustered_genres_2 = pickle.load(f)

models = {
    1: model_1,
    2: model_2,
}

mlbs = {
    1: mlb_1,
    2: mlb_2,
}

saved_genre_to_clusters = {
    1: genre_to_cluster_1,
    2: genre_to_cluster_2,
}

saved_clustered_genres = {
    1: clustered_genres_1,
    2: clustered_genres_2,
}


class Input(BaseModel):
    genres: List[str]
    favourite_genres: List[str]


class ABResult(BaseModel):
    created_at: datetime.datetime
    user_id: int
    group: int
    model: int
    input: Input
    prediction: bool


@app.post("/ab_test")
def ab_test(input: Input, user_id: int):
    group = user_id % 2
    if group == 0:
        prediction = model1_predict(input)["skipped"]
    else:
        prediction = model2_predict(input)["skipped"]

    predictions_collection.insert_one({
        "created_at": pd.Timestamp.now(),
        "user_id": user_id,
        "group": group,
        "model": group + 1,
        "input": input.dict(),
        "prediction": prediction
    })

    return {"skipped": prediction}


@app.get("/ab_test/results")
def ab_test_results() -> List[ABResult]:
    predictions = list(predictions_collection.find({}))
    return predictions


@app.post("/models/{model_id}/predict")
def predict(model_id: int, input: Input):
    if model_id == 1:
        return model1_predict(input)
    elif model_id == 2:
        return model2_predict(input)
    else:
        return {"error": "model not found"}


def model1_predict(input: Input):
    mlb = mlbs[1]
    model = models[1]
    genre_to_cluster = saved_genre_to_clusters[1]
    clustered_genres = saved_clustered_genres[1]

    data = pd.DataFrame([input.dict()])

    # Apply the mapping function to both 'genres' and 'favourite_genres' columns
    data['genres'] = data['genres'].apply(
        lambda x: [map_genre(genre, genre_to_cluster, clustered_genres) for genre in x])
    data['favourite_genres'] = data['favourite_genres'].apply(
        lambda x: [map_genre(genre, genre_to_cluster, clustered_genres) for genre in x])

    encoded_favourite_genres = mlb.transform(data["favourite_genres"])
    encoded_genres = mlb.transform(data["genres"])

    X = np.concatenate([encoded_genres, encoded_favourite_genres], axis=1)

    prediction = model.predict(X)
    print(prediction)
    skipped = bool(prediction[0] == 1)
    print(skipped)

    return {"skipped": skipped}


def model2_predict(input: Input):
    mlb = mlbs[2]
    model = models[2]
    genre_to_cluster = saved_genre_to_clusters[2]
    clustered_genres = saved_clustered_genres[2]

    data = pd.DataFrame([input.dict()])

    data['genres'] = data['genres'].apply(
        lambda x: [map_genre(genre, genre_to_cluster, clustered_genres) for genre in x])
    data['favourite_genres'] = data['favourite_genres'].apply(
        lambda x: [map_genre(genre, genre_to_cluster, clustered_genres) for genre in x])

    encoded_favourite_genres = mlb.transform(data["favourite_genres"])
    encoded_genres = mlb.transform(data["genres"])

    X = np.concatenate([encoded_genres, encoded_favourite_genres], axis=1)

    prediction = model.predict(X)
    print(prediction)
    skipped = bool(prediction[0][0] > 0.5)
    print(skipped)

    return {"skipped": skipped}


def map_genre(genre, genre_to_cluster, clustered_genres):
    cluster_label = genre_to_cluster[genre]
    representative_genre = clustered_genres[cluster_label][0]
    return representative_genre
