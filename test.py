import numpy as np
import reclab
import sys

original_stdout = sys.stdout
env = reclab.make('topics-satiation-v1')
items, users, ratings = env.reset()
with open('sats2.txt', 'w') as f:
    sys.stdout = f
    for i in range(100):
        online_users = env.online_users
        # Recommends 10 random items to each online user.
        recommendations = np.random.choice(
            list(items), size=(len(online_users), 10))
        _, _, ratings, info = env.step(recommendations)
    env.close()
    sys.stdout = original_stdout
