import os
import networkx as nx
import pandas as pd
from pydantic import BaseModel
from typing import Optional


class GraphFiles(BaseModel):
    item_item_graph: nx.DiGraph
    user_item_graph: nx.Graph
    user_item_graph_test: nx.Graph

    @staticmethod
    def from_path(base_path: str) -> "GraphFiles":
        item_item_graph = nx.read_gpickle(os.path.join(base_path, "resources/item_item_graph.p"))
        user_item_graph = nx.read_gpickle(os.path.join(base_path, "resources/user_item_graph.p"))
        user_item_graph_test = nx.read_gpickle(os.path.join(base_path, "resources/user_item_graph_test.p"))
        return GraphFiles(
            item_item_graph=item_item_graph, user_item_graph=user_item_graph, user_item_graph_test=user_item_graph_test
        )

    class Config:
        arbitrary_types_allowed = True


class MovieFiles(BaseModel):
    movies: pd.DataFrame
    rating: Optional[pd.DataFrame]

    @staticmethod
    def from_path(base_path: str) -> "MovieFiles":
        return MovieFiles(movies=pd.read_csv(os.path.join(base_path, "resources/movie.csv")), rating=None)

    class Config:
        arbitrary_types_allowed = True


current_path = os.path.dirname(__file__)
movie_files = MovieFiles.from_path(current_path)
graph_files = GraphFiles.from_path(current_path)
