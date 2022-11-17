import networkx as nx
import pandas as pd
from typing import Generic, Iterator, Tuple, Dict, List, Any
from itertools import permutations
from app.core import BaseEstimator, BaseModel


class Estimator(BaseEstimator):
    def __init__(self, userCol: str = "userId", itemCol: str = "movieId", alpha: float = 0.85) -> None:
        self.userCol = userCol
        self.itemCol = itemCol
        self.alpha = alpha

    def __permutation_generator(self, df: pd.DataFrame) -> Iterator[Tuple[Any, ...]]:
        pairs_generator = permutations(df[self.itemCol].unique(), r=2)
        return pairs_generator

    @staticmethod
    def __get_edge_weight_float(user_item_graph: nx.Graph, item1: str, item2: str) -> float:

        """
        item1 -> item 2 : P(item2|item1) = P(item2 and item1)/P(item1)
        """
        probab_items = len(list(nx.common_neighbors(user_item_graph, item1, item2)))
        return 0.0 if probab_items == 0 else len(list(user_item_graph.neighbors(item1))) / probab_items

    @staticmethod
    def preference_vector(user: int, user_item_graph: nx.Graph) -> Dict[str, float]:
        "we have no weights in user_item graphs because we assume model is ratings agnostic"

        nitems = user_item_graph.degree(user)
        return {item: 1 / nitems for item in user_item_graph.neighbors(user)}

    def build_user_item_bipartite_graph(self, df: pd.DataFrame) -> nx.Graph:
        graph: nx.Graph = nx.Graph()
        graph.add_nodes_from(df[self.userCol].unique(), bipartite=0)
        graph.add_nodes_from(df[self.itemCol].unique(), bipartite=1)

        #  add edges only between nodes of opposite sets
        graph.add_edges_from(list(zip(df[self.userCol], df[self.itemCol])))
        return graph

    def build_item_item_graph(self, df: pd.DataFrame, user_item_graph: nx.Graph) -> nx.DiGraph:
        graph: nx.DiGraph = nx.DiGraph()

        for item1, item2 in self.__permutation_generator(df):
            wt = self.__get_edge_weight_float(user_item_graph, item1, item2)

            if wt:
                graph.add_edge(item1, item2) if wt is True else graph.add_edge(item1, item2, weight=wt)

        return graph

    def get_page_rank_scores(
        self, users: List[int], item_item_graph: nx.DiGraph, user_item_graph: nx.Graph
    ) -> Dict[int, Dict[str, float]]:
        """
        For every user get page rank scores for all movies not watched by the user
        """

        scores = {
            user: {
                movie: score
                for movie, score in nx.pagerank(
                    item_item_graph, personalization=self.preference_vector(user, user_item_graph)
                ).items()
                if movie not in list(user_item_graph.neighbors(user))
            }
            for user in users
        }
        return scores

    def fit(self, df: pd.DataFrame) -> "Model":
        user_item_graph = self.build_user_item_bipartite_graph(df)
        item_item_graph = self.build_item_item_graph(df, user_item_graph)
        model: Model = Model(
            user_item_graph=user_item_graph, item_item_graph=item_item_graph, userCol=self.userCol, itemCol=self.itemCol
        )
        return model


class Model(BaseModel, Estimator):
    def __init__(
        self, user_item_graph: nx.Graph, item_item_graph: nx.DiGraph, userCol: str = "userId", itemCol: str = "movieId"
    ) -> None:
        super(Model, self).__init__(userCol=userCol, itemCol=itemCol)
        self.user_item_graph = user_item_graph
        self.item_item_graph = item_item_graph

    def predict_single_user(self, user: int) -> Dict[str, float]:
        return {
            movie: score
            for movie, score in nx.pagerank(
                self.item_item_graph, personalization=self.preference_vector(user, self.user_item_graph)
            ).items()
            if movie not in list(self.user_item_graph.neighbors(user))
        }

    def __toDf(self, user: int) -> pd.DataFrame:
        score = self.predict_single_user(user)
        return pd.DataFrame([{"userId": user, "movieId": movie, "score": sc} for movie, sc in score.items()])

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        preds = pd.concat([self.__toDf(user) for user in df[self.userCol].unique()])
        return preds.merge(df, on=[self.userCol, self.itemCol])
