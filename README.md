# RecLab
RecLab is a modular simulation environment for online evaluation and comparison recommendation algorithms.


## Geting Started
This section contains a brief guide for getting started with RecLab.

### Basics
The online evaluation of recommender systems consists of two basic components: **User Environments** and **Recommenders**. The environment governs the behavior of users with respect to recommended content. The recommender is using available data from the environment to select items as recommendations. Recommender and environment interact in a step-base fashion: At each time-step the environment specifies a set of users that are available to receive recommendations (we call them _online users_), along with any other side information. The recommender uses the history of interactions to recommend a single item (Top-1 recommendation), or a set of items (slate based recommendation) to each online user. In turn the environment respond by providing ratings to recommended items in Top-1 setting or by selecting a item form a slate of potential items in a Slate based setting.

Below is a visualization of the interaction between environment and recommender.

![Flowchart](/figures/RecLab3.pdf)

#### Environments
The basic interface for an environment that all environments inherit from is [Environment](reclab/environments/environment.py). The most important methods in developing a new environments are:

- `reset`: method that resets the environment to its original state. Must be called before the first step of the simulation.
- `online_users`: method that returns a list of available users from the environment.
- `step(recommendations)`: main method that environments must implement. It takes in the `recommendations` from the recommender, updates the internal state of the environment and returns the following, which are in turn passed to the recommender:
    - `users`: New users and users whose information got updated this timestep, along with any side information about each user.
    - `items`: New items and items whose information got updated this timestep, along with any side information about each item.
    - `ratings`: New ratings and ratings whose information got updated this timestep, along with any side information about each rating.
    - `info`: Extra information that can be used for debugging but should not be made accessible to the recommender.


To see a description of available environments see the [List of Enviroments](reclab/environments/README.md).

#### Recommenders


#### Recommenders

## Setup
Get started with RecLab by cloning the repository.

**Coming soon**: pip installable RecLab package.

RecLab was developed and tested Python version 3.8.

### Requirements
We suggest installing RecLab in a virtual environment and installing dependencies from [requirements.txt](requirements.txt).

## Metrics

### Running Experiments
