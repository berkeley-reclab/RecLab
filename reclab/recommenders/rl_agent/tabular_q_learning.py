import numpy as np

class TabularQLearning(object):
    """ Simplest reinforcement agent based recommender.
    It uses tabular Q-learning to learn a Q-value table 
    for all user-topic pairs. Q : SxA -> R

    S: users
    A: topics

    The algorithm starts with Q-values derived from initial ratings
    At each iteration the algorithm is presented with a fraction of
    online users. For those users it selects the action with the 
    largest q value with probability epsilon and a random action with
    probability 1-epsilon. Once a topic is selected, the algorithm chooses
    an arbitrary item that belongs to that topic.

    The recommender receives back a reward, the simplest form of reward is
    just the user rating of that item. The reward signal is used to update the
    Q-values of the given topic. Ultimately the goal of this algorithm is to
    discover hidden topic preference model from interactions with the users
    
    """    

    def __init__(self, max_num_users, max_num_items, max_num_topics,
                    learning_rate = 0.1, discount_factor = 0.9, epsilon = 0.9):
        self._users = {}
        self._max_num_users = max_num_users
        self._items = {}
        self._max_num_items = max_num_items
        self._rated_items = {}
        self._topics = {}
        self._max_num_topics = max_num_topics

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor  
        self.epsilon = epsilon

        self.QTable = np.empty([self._max_num_users, self._max_num_topics])
        
    def init(self, items, users, ratings):  
        pass

    def update(self, items, users, ratings);
        pass

    def recommend(self, user, item):
        pass     
  
    def clear(self):
        pass