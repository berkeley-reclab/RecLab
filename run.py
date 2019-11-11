import numpy as np

from reclab.environments.topics import Topics
from reclab.environments.repeat_topics import RepeatTopics
from reclab.recommenders.libfm.libfm import LibFM


def main():
    env = Topics(num_topics=10, num_users=100, num_items=1700, num_init_ratings=10000)
    # env = RepeatTopics(num_topics=10, num_users=100, num_items=1700, num_init_ratings=10000)
    recommender = LibFM(num_user_features=0, num_item_features=0, num_rating_features=0, max_num_users=100000, max_num_items=100000)

    # First generate the items and users to seed the dataset.
    print("Initializing environment and recommender")
    items, users, ratings = env.reset()
    recommender.reset(items, users, ratings)

    # Now recommend items to users.
    print("Making online recommendations")
    for i in range(1):
        online_users = env.online_users()
        ret, predicted_ratings = recommender.recommend(online_users, num_recommendations=1)
        recommendations = ret[:, 0]
        items, users, ratings, info = env.step(recommendations)
        recommender.update(users, items, ratings)
        errors = ratings[:,2] - predicted_ratings[:,0]
        print("AAAAAA", i, np.mean(ratings[:, -1]), np.mean(errors**2))

    ratings = env.all_ratings()
    print(np.mean(ratings))
    print(np.std(ratings))
main()
