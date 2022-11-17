from typing import Dict, Callable, Tuple, Any, List, Optional
import numpy as np
import pandas as pd


class QueryMetrics:
    """
    Base class that calculates the given metric for all the queries present in the dataframe. The class takes a \
    dataframe. Every row of the dataframe is a document which must have a query id, relevance score and a predicted \
    score. These three column names can be set in the constructor of the derived class. The output of the class is a \
    dictionary with keys as query ids and values as metric score.

    :param user_col: name of the column of the dataframe having query ids
    :param preds_col: name of the column of the dataframe having predicted scores
    :param relevance_col: name of the column of the dataframe having relevance scores

    """

    def __init__(self, user_col: str, preds_col: str, relevance_col: str) -> None:
        self.user_col = user_col
        self.preds_col = preds_col
        self.relevance_col = relevance_col

    def __totuple__(self, x: pd.DataFrame) -> List[Tuple[Any, Any]]:
        return list(zip(x[self.relevance_col], x[self.preds_col]))

    @staticmethod
    def _sortByscore(x: List[Tuple[Any, Any]]) -> List[Tuple[Any, Any]]:
        return sorted(x, key=lambda k: k[1], reverse=True)  # type: ignore

    @staticmethod
    def _sortByrelevance(x: List[Tuple[Any, Any]]) -> List[Tuple[Any, Any]]:
        return sorted(x, key=lambda k: k[0], reverse=True)  # type: ignore


class Ndcg(QueryMetrics):
    """
    This class implements the Normalized Discounted Cumulative Gain metric. Ndcg is a metric widely used in information
    retrieval domain to evaluate quality of search results of a search engine. Since we care more about top ranking \
    documents rather than the bottom ranking documents, this metric penalizes more if ranking results have mistakes \
    in the top order compared to mistakes in the bottom order. The range of this metric is between 0-1 and a higher \
    score is better.

    This class assumes top ranking documents should have a higher relevance score. If there are 5 documents in the best
    ranking order 1,2,3,4,5 then the relevance scores should be 5,4,3,2,1.

    The relevance scores can be scaled using the label_gain parameter which is a dictionary. For example to scale the \
    relevances 5,4,3,2,1 by a factor of 10 we can set label_gain={1:10,2:20,3:30,4:40,5:50} meaning the 1st ranking \
    document which has a relevance of 5 is now weighted as 50 and the least ranking document is weighted as 10.

    For a single query_id the metric can be calculated by calling Ndcg.get_metric method. The input should be a list \
    of tuples where every tuple is made of (relevance, score) where relevance is the relevance of the document and \
    score is the score of the document assigned by the estimator. The output is a score in float of the given query id.

    For multiple query ids the metric can be calculated by calling Ndcg.calculate_metrics which is a method of it's \
    baseclass. The method takes a dataframe having query id, relevance and predicted score columns. These columns can \
    be set in the constructor of the class. The output is a dictionary with keys as query ids and values as metric \
    score.

    :param k: integer specifying topk documents to consider when calculating ndcg score.
    :param label_gain: dictionary with keys as relevance and values as corresponding weight for relevance.
    :param _log: function to calculate logarithm. Default is np.log2 which is log base 2.
    :param user_col: name of the column of the dataframe having query ids
    :param preds_col: name of the column of the dataframe having predicted scores
    :param relevance_col: name of the column of the dataframe having relevance scores

    Usage:

    df = pd.DataFrame({'query_id':[ii for i in range(5) for ii in [i]*5],
                   'preds':[ii for i in range(5) for ii in np.random.random_sample((5,))],
                   'relevance':[ii for i in range(5) for ii in np.random.choice(range(1,6),size=5, replace=False)]})

    out = Ndcg(k=5,  user_col='query_id', preds_col='preds', relevance_col='relevance').calculate_metrics(df)

    #out
    {0: 0.9542080637541479,
     1: 0.7392561108165644,
     2: 0.7875305117652345,
     3: 0.9073155929493236,
     4: 0.8386893446302166}

    """

    def __init__(
        self,
        k: int,
        label_gain: Optional[Dict[int, float]] = None,
        _log: Callable = np.log2,
        user_col: str = "userId",
        preds_col: str = "score",
        relevance_col: str = "rating",
    ) -> None:
        super(Ndcg, self).__init__(user_col=user_col, preds_col=preds_col, relevance_col=relevance_col)
        self.k = k
        self.label_gain = label_gain if label_gain is not None else {i: i for i in range(32)}
        self._log = _log

    def __getWeight__(self, rel: int) -> float:
        return self.label_gain.get(rel, rel)

    def dcg(self, relevance_with_scores: List[Tuple[Any, Any]]) -> float:
        relevances = [self.__getWeight__(rel) for rel, score in self._sortByscore(relevance_with_scores)[: self.k]]
        indexes = np.arange(1, len(relevances) + 1)
        num: float = np.sum(relevances / self._log(indexes + 1))
        return num

    def idcg(self, relevance_with_scores: List[Tuple[Any, Any]]) -> float:
        relevances = [self.__getWeight__(rel) for rel, score in self._sortByrelevance(relevance_with_scores)[: self.k]]
        indexes = np.arange(1, len(relevances) + 1)
        num: float = np.sum(relevances / self._log(indexes + 1))
        return num

    def get_metric(self, relevance_with_scores: List[Tuple[Any, Any]]) -> float:
        return self.dcg(relevance_with_scores) / self.idcg(relevance_with_scores)

    def calculate_metrics(self, df: pd.DataFrame) -> Dict[Any, float]:
        query_dict = {
            query_id: self.__totuple__(df[df.index.isin(q_index)])
            for query_id, q_index in df.groupby([self.user_col]).groups.items()
        }

        metric_dict = {query_id: self.get_metric(xx) for query_id, xx in query_dict.items()}
        return metric_dict
