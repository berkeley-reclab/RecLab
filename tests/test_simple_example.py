"""
Test the basic example found in the README
"""
import numpy as np
import reclab


def test_basic_example():
    """Test the basic example in the README."""
    env = reclab.make('topics-dynamic-v1')
    items, users, ratings = env.reset()
    for i in range(1):
        online_users = env.online_users
        # Your recommendation algorithm here. This recommends 10 random items to each online user.
        recommendations = np.random.choice(list(items), size=(len(online_users), 10))
        _, _, ratings, info = env.step(recommendations)
    env.close()
