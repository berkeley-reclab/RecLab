"""A set of recommender to be used in conjunction with environments."""
from .knn_recommender import KNNRecommender
from .libfm import LibFM
from .top_pop import TopPop
from .baseline import RandomRec
from .baseline import PerfectRec

try:
    from .autorec import Autorec
    from .cfnade import Cfnade
    from .llorma import Llorma
    from .sparse import SLIM, EASE
except ImportError:
    pass
