"""A set of recommender to be used in conjunction with environments."""
from .autorec import Autorec
try:
	from .cfnade import Cfnade
except ImportError as e:
	print('Unable to import CFNade')
from .knn_recommender import KNNRecommender
from .libfm import LibFM
from .llorma import Llorma
from .sparse import SLIM, EASE
from .top_pop import TopPop
from .baseline import RandomRec
from .baseline import PerfectRec
