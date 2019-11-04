import numpy as np

from reclab.environments.fixed_rating import FixedRating
from reclab.recommenders.libfm.libfm import LibFM


def main():
    env = FixedRating(num_users=100, num_items=170, num_init_ratings=5000, rating_frequency=0.2)
    recommender = LibFM(num_user_features=0, num_item_features=0, num_rating_features=0, max_num_users=100000, max_num_items=100000)

    # First generate the items and users to seed the dataset.
    items, users, ratings = env.reset()
    recommender.init(items, users, ratings)

    # Now recommend items to users.
    for i in range(100):
        online_users = env.online_users()
        recommendations = recommender.recommend(online_users, num_recommendations=1)[:, 0]
        items, users, ratings, info = env.step(recommendations)
        recommender.update(users, items, ratings)
        print("AAAAAA", np.mean(ratings[:, -1]))

    ratings = env.get_all_ratings()
    print(ratings)
main()
