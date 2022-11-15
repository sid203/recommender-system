import os
import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Any, Optional, Tuple


def preference_vector(user: int, user_item_graph: nx.Graph) -> Dict[str, float]:

    "we have no weights in user_item graphs because we assume model is ratings agnostic"

    nitems = user_item_graph.degree(user)

    return {item: 1 / nitems for item in user_item_graph.neighbors(user)}


def get_page_rank_scores(
    users: List[int], item_item_graph: nx.DiGraph, user_item_graph: nx.Graph
) -> Dict[int, Dict[str, float]]:
    scores = {
        user: {
            movie: score
            for movie, score in nx.pagerank(
                item_item_graph, personalization=preference_vector(user, user_item_graph)
            ).items()
            if movie not in list(user_item_graph.neighbors(user))
        }
        for user in users
    }
    return scores


def top_recommendations(results: Dict, k: int, mapping_file: pd.DataFrame) -> None:

    userid: int = list(results)[0]
    top_movies: List[Tuple] = sorted(results[userid].items(), key=lambda x: x[1], reverse=True)[:k]  # type: ignore

    title_mapping = dict(zip(mapping_file.movieId, mapping_file.title))
    genre_mapping = dict(zip(mapping_file.movieId, mapping_file.genres))

    for movieid, score in top_movies:
        print(
            f"movieid:{int(movieid)} movie title:{title_mapping.get(int(movieid), '')}"
            f" movie genre:{genre_mapping.get(int(movieid))} score:{round(score, 6)}"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Argument parser for running inference")

    parser.add_argument(
        "--userid",
        type=int,
        default=55420,
        help="User ids from MovieLens dataset occuring in the subsample. "
        "Can be one of [55420, 7051, 40499, 91897, 136751, 54494, 127235, 66667, 24006, 134898]",
    )

    parser.add_argument("--topk", type=int, default=5, help="Number of top recommendations sought for the given user")

    args = parser.parse_args()

    ROOT: str = os.getenv("ROOT_FOLDER", default="")
    item_item_graph = nx.read_gpickle(os.path.join(ROOT, "app/resources/item_item_graph.p"))
    user_item_graph = nx.read_gpickle(os.path.join(ROOT, "app/resources/user_item_graph.p"))
    user_item_graph_test = nx.read_gpickle(os.path.join(ROOT, "app/resources/user_item_graph_test.p"))
    movies = pd.read_csv(os.path.join(ROOT, "app/resources/movie.csv"))

    nn = [node for node, bipartite in user_item_graph.nodes(data="bipartite") if bipartite == 0]

    r = get_page_rank_scores([args.userid], item_item_graph, user_item_graph)
    top_recommendations(r, args.topk, movies)
