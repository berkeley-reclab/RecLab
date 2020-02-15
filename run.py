import numpy as np

from reclab.environments.latent_factors import LatentFactorBehavior
from reclab.environments.topics import Topics
from reclab.recommenders.libfm.libfm import LibFM
from reclab.recommenders import TopPop
from reclab.recommenders import KNNRecommender


def main():
    params = {'topic_change': 0.1, 'memory_length': 5, 
              'boredom_threshold': 2, 'boredom_penalty': 1.0}
    env = Topics(num_topics=10, num_users=100, num_items=170, num_init_ratings=5000, **params)
    # params = {'affinity_change': 0.1, 'memory_length': 5, 
    #           'boredom_threshold': 0.5, 'boredom_penalty': 1.0}
    # env = LatentFactorBehavior(latent_dim=8, num_users=100, num_items=170, num_init_ratings=1000, **params)
    # env = MovieLens100k(latent_dim=8, datapath="~/recsys/data/ml-100k/", num_init_ratings=1000)
    # env = RandomPreferences(num_topics=10, num_users=100, num_items=1700, num_init_ratings=10000)
    recommender = KNNRecommender() # TopPop() # LibFM(num_user_features=0, num_item_features=0, num_rating_features=0, max_num_users=100, max_num_items=170)

    # First generate the items and users to seed the dataset.
    print("Initializing environment and recommender")
    items, users, ratings = env.reset()
    recommender.reset(items, users, ratings)

    # Now recommend items to users.
    print("Making online recommendations")
    for i in range(100):
        online_users = env.online_users()
        ret, predicted_ratings = recommender.recommend(online_users, num_recommendations=1)
        recommendations = ret[:, 0]
        items, users, ratings, info = env.step(recommendations)
        recommender.update(users, items, ratings)
        rating_arr = []
        print(predicted_ratings)
        if predicted_ratings is not None:
            for (rating, _), pred in zip(ratings.values(), predicted_ratings):
                rating_arr.append([rating, pred])
            rating_arr = np.array(rating_arr)
            errors = rating_arr[:,0] - rating_arr[:,1]
            print("Iter:", i, "Mean:", np.mean(rating_arr[:, 0]), "MSE:", np.mean(errors**2))
        else:
            for (rating, _) in ratings.values():
                rating_arr.append(rating)
            rating_arr = np.array(rating_arr)
            print("Iter:", i, "Mean:", np.mean(rating_arr))

    ratings = env.all_ratings()
main()
