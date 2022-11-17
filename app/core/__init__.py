from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Sequence
import pandas as pd
import networkx as nx

D = TypeVar("D", bound=pd.DataFrame)  # dataframe
G = TypeVar("G", bound=nx.Graph)  # undirected graph
DG = TypeVar("DG", bound=nx.DiGraph)  # directed graph
M = TypeVar("M")  # model


class Transformer(ABC, Generic[D]):
    @abstractmethod
    def apply(self, df: D) -> D:
        pass


class SequenceTransformer(ABC, Generic[D]):
    @abstractmethod
    def split(self, df: D) -> Sequence[D]:
        pass


class BaseEstimator(ABC, Generic[D, G, DG, M]):
    @abstractmethod
    def fit(self, df: D) -> M:
        pass


class BaseModel(ABC):
    @abstractmethod
    def predict(self, df: D) -> D:
        pass
