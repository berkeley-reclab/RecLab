"""
Test the basic example found in the README
"""
import numpy as np
import reclab


def test_basic_example():
    """
    Test the basic example in the READMe
    """
    n_users = 1000
    n_topics = 10
    n_items = 20
    env = reclab.make(
        'topics-dynamic-v1',
        num_topics=n_topics,
        num_users=n_users,
        num_items=n_items,
        num_init_ratings=n_users * n_topics,
    )
    users, items, _ = env.reset()
    assert len(users) == n_users
    assert len(items) == n_items
    for _ in range(1):
        online_users = env.online_users
        # Your recommendation algorithm here.
        # This recommends 2 random items to each online user.
        recommendations = np.random.choice(
            len(items), size=(len(online_users), 2)
        )
        users, items, _, _ = env.step(recommendations)

    env.close()
