import numpy as np

from reclab.environments.simple import Simple
from reclab.environments.latent_factors import LatentFactorBehavior, MovieLens100k
from reclab.recommenders.libfm.libfm import LibFM


def main():
    # env = LatentFactorBehavior(latent_dim=8, num_users=100, num_items=170, num_init_ratings=1000)
    env = MovieLens100k(latent_dim=8, datapath="~/recsys/data/ml-100k/", num_init_ratings=1000)
    # env = Simple(num_topics=8, num_users=100, num_items=170, num_init_ratings=1000)
    recommender = LibFM(num_user_features=0, num_item_features=0, num_rating_features=0, max_num_users=100, max_num_items=170)

    # First generate the items and users to seed the dataset.
    print("Initializing environment and recommender")
    items, users, ratings = env.reset()
    recommender.init(items, users, ratings)

    # Now recommend items to users.
    print("Making online recommendations")
    for i in range(10):
        online_users = env.online_users()
        ret, predicted_ratings = recommender.recommend(online_users, num_recommendations=1)
        recommendations = ret[:, 0]
        items, users, ratings, info = env.step(recommendations)
        recommender.update(users, items, ratings)
        errors = ratings[:,2] - predicted_ratings[:,0]
        print("AAAAAA", i, np.mean(ratings[:, -1]), np.mean(errors**2))

    ratings = env.all_ratings()
    print(np.mean(ratings))
main()
