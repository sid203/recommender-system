from typing import Sequence, List
import pandas as pd
import numpy as np
from app.core.utils import compose_with_args
from sklearn.model_selection import train_test_split
from app.core import Transformer, SequenceTransformer


class SubSampler(Transformer):
    def __init__(
        self,
        userCol: str = "userId",
        itemCol: str = "movieId",
        size: float = 0.10,
        frequency_thresh: int = 10,
        seed: int = 42,
    ) -> None:
        self.userCol = userCol
        self.itemCol = itemCol
        self.size = size
        self.seed = seed
        self.thresh = frequency_thresh
        np.random.seed(seed)

    @staticmethod
    def __frequency_filter(df: pd.DataFrame, col: str, thresh: int) -> pd.DataFrame:
        m = df[col].value_counts() > thresh
        res: pd.DataFrame = df[df[col].isin(m[m].index)]
        return res

    @staticmethod
    def __sub_sample(df: pd.DataFrame, col: str, size: float) -> pd.DataFrame:
        uniques = np.unique(df[col]) if col else np.unique(df.index)
        subsample = np.random.choice(uniques, size=int(len(uniques) * size), replace=False)
        res: pd.DataFrame = df[df[col].isin(subsample)] if col else df[df.index.isin(subsample)]
        return res

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:

        fs_with_args = [
            (self.__sub_sample, [self.userCol, self.size]),
            (self.__sub_sample, [self.itemCol, self.size]),
            (self.__frequency_filter, [self.itemCol, self.thresh]),
            (self.__frequency_filter, [self.userCol, self.thresh]),
        ]

        out: pd.DataFrame = compose_with_args(fs_with_args, df)
        return out


class Splitter(SequenceTransformer):
    def __init__(self, test_size: float = 0.2, itemCol: str = "movieId", seed: int = 42) -> None:
        self.test_size = test_size
        self.itemCol = itemCol
        self.seed = seed

    def split(self, df: pd.DataFrame) -> Sequence[pd.DataFrame]:
        train, test = train_test_split(df, test_size=self.test_size, stratify=df[self.itemCol], random_state=self.seed)
        return train, test


class ColumnsDropper(Transformer):
    def __init__(self, cols_to_drop: List[str]) -> None:
        self.cols_to_drop = cols_to_drop

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        res: pd.DataFrame = df.drop(columns=self.cols_to_drop)
        return res
