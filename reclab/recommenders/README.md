## List of Recommenders

All provided recommenders are subclasses of `PredictRecommender`, which uses rating predictions to make recommendations. It supports both deterministic and stochastic item selection policies.

### Baseline Recommenders

#### [RandomRec](reclab/recommenders/baseline.py)
A recommender that returns a random item from the list of unconsumed items foreach online user. It is a useful baseline for calibrating lower-bounds of recommender performance.

#### [TopPop](reclab/recommenders/top_pop.py)
A recommender that uses historical ratings to make global rankings of items and recommend items based on items with highest overall popularity. This is a useful baseline for measuring the benefits of personalization.

#### [PerfectRec](reclab/recommenders/baseline.py)
A recommender that is instantiated with a `dense_rating_function`, which provides the true ratings of the users for all items. It is a useful baseline for calibrating upper-bounds of recommender performance.

### [Neighborhood-based recommenders](reclab/recommenders/knn_recommender.py)
`KNNRecommender` neighborhood based collaborative filtering algorithm. The class supports both user and item based collaborative filtering. In an `user_based` KNN recommender, user features are stacked and pairwise similarity metrics between users are measured. An online user is thus recommended an item that was highly rated by a similar user. Conversely in an  `item_based` KNN recommender, item features are stacked and pairwise similarity metrics between items are measured. An online user is thus recommended an item that is highly similar to other highly rated items of the user.

### [Matrix Factorization](reclab/recommenders/libfm.py)
It is wrapper for the [LibFM recommender](https://www.csie.ntu.edu.tw/~b97053/paper/Factorization%20Machines%20with%20libFM.pdf). See www.libfm.org for implementation details. We built a pip installable python package **`ypyfm`** based on [this C++ implementation](https://github.com/srendle/libfm), that might be of interest in it's own right.

At each step of the simulation the `LibFM` recommender re-trains a matrix factorization model. It computes rating predictions as the inner product of user and item factors plus bias terms.

### [AutoRec](reclab/recommenders/autorec/autorec.py)
`AutoRec` is an autoencoder framework for collaborative filtering proposed by this [paper](https://dl.acm.org/doi/10.1145/2740908.2742726). It can be seen as a non-linear generalization of factorization models. We adapted a publicly available [implementation](https://github.com/mesuvash/NNRec)

### [CF-NADE](/reclab/recommenders/cfnade/cfnade.py)
`Cfnade` is neural autoregressive architecture for collaborative filtering proposed by this [paper](https://arxiv.org/pdf/1605.09477.pdf). We adapted a publicly available [implementation](https://github.com/JoonyoungYi/CFNADE-keras).

### [LLORMA](reclab/recommenders/llorma/llorma.py)
`Llorma` is a generalization of low rank matrix factorization techniques based on this (paper)[http://jmlr.org/papers/v17/14-301.html]. The LLORMA algorithm approximates the observed rating matrix as a weighted sum of low-rank matrices which are limited to a local region of the observed matrix. We adapted a publicly available [implementation](https://github.com/JoonyoungYi/LLORMA-tensorflow).

### [Sparse Recommenders](reclab/recommenders/sparse.py)
`SLIM` is a sparse linear recommendation model based on this [paper](http://glaros.dtc.umn.edu/gkhome/node/774). For an user *i* it models the predicted rating of an unseen item *i* as a weighted average of the ratings of items previously rated by user *u*.

`EASE` predicts ratings bases on item-item similarity model based on this [paper](https://arxiv.org/pdf/1905.03375.pdf). Assuming that the historical data contains *N* users and *M* items in a *NxM* rating matrix *X*. The model computes a *MxM* self-similarity matrix *B*. Unseen ratings are predicted as *XB*.
