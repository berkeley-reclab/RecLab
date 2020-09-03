# RecLab
RecLab is a modular simulation environment for online evaluation and comparison of recommendation algorithms.


## Geting Started
This section contains a brief guide for getting started with RecLab.

### Basics
The online evaluation of recommender systems consists of two basic components: **User Environments** and **Recommenders**. The environment governs the behavior of users with respect to recommended content. The recommender is using available data from the environment to select items as recommendations. Recommender and environment interact in a step-base fashion: At each time-step the environment specifies a set of users that are available to receive recommendations (we call them _online users_), along with any other side information. The recommender uses the history of interactions to recommend a single item (Top-1 recommendation), or a set of items (slate based recommendation) to each online user. In turn the environment respond by providing ratings to recommended items in Top-1 setting or by selecting a item form a slate of potential items in a Slate based setting.

Below is a visualization of the interaction between environment and recommender.

![Flowchart](/figures/RecSys.png)

#### Environments
The basic interface for an environment that all environments inherit from is [Environment](reclab/environments/environment.py). The most important methods in developing a new environment are:

- `reset`: method that resets the environment to its original state. Must be called before the first step of the simulation.
- `online_users`: method that returns a list of available users from the environment.
- `step(recommendations)`: main method that environments must implement. It takes in the `recommendations` from the recommender, updates the internal state of the environment and returns the following, which are in turn passed to the recommender:
    - `users`: New users and users whose information got updated this timestep, along with any side information about each user.
    - `items`: New items and items whose information got updated this timestep, along with any side information about each item.
    - `ratings`: New ratings and ratings whose information got updated this timestep, along with any side information about each rating.
    - `info`: Extra information that can be used for debugging but should not be made accessible to the recommender.


To see a description of available environments see the [List of Enviroments](reclab/environments/README.md).

#### Recommenders
The basic interface for a recommender that all recommenders inherit from is [Recommender](reclab/recommenders/recommender.py). The most important methods for adding a new recommender are:

- `recommend(user_contexts, num_recommendations)`: Method that returns a list of items to be recommended to each online user. The `user_contexts` contains all the information about online users provided by the environment.
- `update(users, items, ratings)`: Method that updates the recommender at each time-step with the new `user`, `item` and `rating` data provided by the environment.

To see a description of available recommenders see the [List of Recommenders](reclab/recommenders/README.md).



## Setup
RecLab was developed and tested Python version 3.8. Get started with RecLab by cloning the repository.

**Coming soon**: pip installable RecLab package.

### Requirements
We suggest installing RecLab in a virtual environment and installing dependencies from [requirements.txt](requirements.txt).

### Running Experiments
See below a simple usage example:
```
import numpy as np
import reclab
env = reclab.make('topics-dynamic-v1')
items, users, ratings = env.reset()
for _ in range(1000):
    online_users = env.online_users()
    # Your recommendation algorithm here. This recommends 10 random items to each online user.
    recommendations = np.random.choice(items, size=(len(online_users), 10))
    items, users, ratings, info = env.step(recommendations)
env.close()
```

**Coming soon:** More functionality for running experiments and custom performance metrics.
