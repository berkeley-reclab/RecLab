"""A set of recommender to be used in conjunction with environments."""
from .autorec import Autorec
from .knn_recommender import KNNRecommender
from .libfm import LibFM
from .llorma import Llorma
from .sparse import SLIM, EASE
from .top_pop import TopPop
from .baseline import RandomRec
from .baseline import PerfectRec
