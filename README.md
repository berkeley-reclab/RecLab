![Build Status](https://travis-ci.com/berkeley-reclab/RecLab.svg?branch=master)
[![Coverage Status](https://coveralls.io/repos/github/berkeley-reclab/RecLab/badge.svg?branch=master)](https://coveralls.io/github/berkeley-reclab/RecLab?branch=master)

# RecLab
RecLab is a simulation framework used to evaluate recommendation algorithms. The framework makes
no platform-specific assumptions. As such, it can be used to evaluate recommendation algorithms
implemented with any computational library.

Reclab is under active development. If you find a bug or would like to request a new feature
please file an [issue](https://github.com/berkeley-reclab/reclab/issues). Furthermore, we welcome a
broad set of contributions including: documentation, tests, new environments, reproduced
recommenders, and code quality improvements. Simply fork the repo and make a
[pull request](https://github.com/berkeley-reclab/reclab/pulls).

## Getting Started
This section contains a brief guide on how to get started with RecLab.

### Setup
RecLab was developed and tested in Python 3.8. To install RecLab run
```
pip install reclab
```
RecLab also implements a set of benchmark recommender systems, however the default
`pip install` command will not fetch the necessary dependencies. To fetch these dependencies
you must have g++ 5.0 or higher and [python3-dev](https://stackoverflow.com/a/21530768)
installed. You should then run
```
pip install reclab[recommenders]
```
which will install both the core reclab framework and the benchmark recommendation algorithms.

### Example
The code below shows a simple use-case with random recommendations.
```python
import numpy as np
import reclab
env = reclab.make('topics-dynamic-v1')
items, users, ratings = env.reset()
for i in range(1000):
    online_users = env.online_users
    # Your recommendation algorithm here. This recommends 10 random items to each online user.
    recommendations = np.random.choice(list(items), size=(len(online_users), 10))
    _, _, ratings, info = env.step(recommendations)
env.close()
```

## RecLab Design
This section briefly outlines the overall design of RecLab, and how to add new environments.

### Basics
Evaluation in RecLab consists of two basic components: **Environments** and **Recommenders**.
An environment consists of a set of users and items. A recommender and an environment interact
iteratively. At each time-step the environment specifies a set of _online users_ that need to be
recommended an item. The recommender uses the history of user-item interactions to either recommend
a single item (top-1 recommendation), or a set of items (slate-based recommendation) to each online
user. The environment then provides ratings to some of, or all, the recommended items.

Below is a visualization of the interaction between environment and recommender.

![Flowchart](/figures/RecSys.png)

#### Environments
In RecLab all environments inherit from the [`Environment`](reclab/environments/environment.py) interface. The following methods must be implemented:
- `reset`: Reset the environment to its original state. Must be called before the first step of the simulation.
- `online_users`: Return a list of available users at each timestep.
- `step(recommendations)`: Given `recommendations`, update the internal state of the environment and return the following data:
    - `users`: New users and users whose information got updated this timestep, along with any side information about each user.
    - `items`: New items and items whose information got updated this timestep, along with any side information about each item.
    - `ratings`: New ratings and ratings whose information got updated this timestep, along with any side information about each rating.
    - `info`: Extra information that can be used for debugging but should not be made accessible to the recommender.

To see a description of available environments see the [list of enviroments](reclab/environments/README.md).

#### Recommenders
RecLab does not assume recommendation algorithms are implemented in any specific way. However, we
also provide a [convenient interface](reclab/recommenders/recommender.py) to simplify the design of
new recommendation algorithms.

To see a description of available recommenders see the
[list of recommenders](reclab/recommenders/README.md). Note that you must install the optional
dependencies to use some of these recommenders as outline under the [setup section](#Setup).

**Coming soon:** More functionality for running experiments and custom performance metrics.
