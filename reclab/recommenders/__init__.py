"""A set of recommender to be used in conjunction with environments."""
from .baseline import RandomRec
from .baseline import PerfectRec
from .knn_recommender import KNNRecommender
from .recommender import Recommender
from .recommender import PredictRecommender
from .top_pop import TopPop

try:
    from .autorec import Autorec
    from .cfnade import Cfnade
    from .libfm import LibFM
    from .llorma import Llorma
    from .sparse import SLIM, EASE
except ImportError:
    pass
