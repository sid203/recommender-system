from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Tuple

from app.core.estimator import Model
from app.files import graph_files, movie_files

app = FastAPI()


class MovieResponseModel(BaseModel):
    id: int
    title: str
    genre: str
    score: float


class UserPayload(BaseModel):
    userid: int
    k: int = 10


# endpoints


@app.post(
    "/predict",
    response_model=List[MovieResponseModel],
    status_code=200,
    summary="Obtain predictions for a given userid",
    description="Sample user ids [55420, 7051, 40499, 91897, 136751, 54494, 127235, 66667, 24006, 134898]",
)
def get_prediction(payload: UserPayload) -> List[MovieResponseModel]:

    userid = payload.userid
    topk = payload.k
    model = Model(user_item_graph=graph_files.user_item_graph, item_item_graph=graph_files.item_item_graph)

    score = model.predict_single_user(userid)
    top_movies: List[Tuple] = sorted(score.items(), key=lambda x: x[1], reverse=True)[:topk]  # type: ignore

    title_mapping = dict(zip(movie_files.movies.movieId, movie_files.movies.title))
    genre_mapping = dict(zip(movie_files.movies.movieId, movie_files.movies.genres))

    response = [
        MovieResponseModel(
            id=int(movieid),
            title=title_mapping.get(int(movieid), ""),
            genre=genre_mapping.get(int(movieid)),
            score=score,
        )
        for movieid, score in top_movies
    ]
    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", port=8080, reload=True)
