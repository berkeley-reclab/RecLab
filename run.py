import numpy as np
import time

from reclab.environments.fixed_rating import FixedRating
from reclab.environments.latent_factors import DatasetLatentFactor
from reclab.environments.topics import Topics
from reclab.recommenders import LibFM
from reclab.recommenders import TopPop
from reclab.recommenders import KNNRecommender
from reclab.recommenders.autorec.autorec import Autorec
from reclab.recommenders.llorma.llorma import Llorma
from reclab.recommenders import KNNRecommender

def main():
    params = {'topic_change': 0.1, 'memory_length': 5, 'rating_frequency': 0.2,
             'boredom_threshold': 2, 'boredom_penalty': 1.0}
    env = Topics(num_topics=5, num_users=100, num_items=170, num_init_ratings=1000, **params)
    # params = {'affinity_change': 0.1, 'memory_length': 5,
    #           'boredom_threshold': 0.5, 'boredom_penalty': 1.0}
    # env = LatentFactorBehavior(latent_dim=8, num_users=100, num_items=170, num_init_ratings=1000, **params)
    # env = DatasetLatentFactor('lastfm', num_init_ratings=1000, max_num_users=100, max_num_items=170)
    # env = RandomPreferences(num_topics=10, num_users=100, num_items=1700, num_init_ratings=10000)
    # recommender = TopPop()
    # recommender = LibFM(num_user_features=0, num_item_features=0, num_rating_features=0, max_num_users=1000, max_num_items=1700)
    recommender = Llorma(max_user=100, max_item=170, use_cache=True)

    # First generate the items and users to seed the dataset.
    print("Initializing environment and recommender")
    users, items, ratings = env.reset()
    recommender.reset(users, items, ratings)

    # Now recommend items to users.
    print("Making online recommendations")
    for i in range(100):
        online_users = env.online_users()
        start_time = time.time()
        ret, predicted_ratings = recommender.recommend(online_users, num_recommendations=1)
        recommendations = ret[:, 0]
        users, items, ratings, info = env.step(recommendations)
        recommender.update(users, items, ratings)
        stop_time = time.time()
        print('Run time: {:.2f}'.format(stop_time-start_time))
        rating_arr = []
        if predicted_ratings is not None:
            for (rating, _), pred in zip(ratings.values(), predicted_ratings):
                rating_arr.append([rating, pred])
            rating_arr = np.array(rating_arr)
            errors = np.abs(rating_arr[:,0] - rating_arr[:,1])
            print("Iter:", i, "Mean:", np.mean(rating_arr[:, 0]), "RMSE:", np.mean(errors))
        else:
            for (rating, _) in ratings.values():
                rating_arr.append(rating)
            rating_arr = np.array(rating_arr)
            print("Iter:", i, "Mean:", np.mean(rating_arr))

    ratings = env.all_ratings()
main()