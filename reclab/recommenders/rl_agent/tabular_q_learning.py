import numpy as np

class TabularQLearning(object):
    """ Simplest reinforcement agent based recommender.
    It uses tabular Q-learning to learn a Q-value table 
    for all user-topic pairs. Q : SxA -> R

    S: state, users
    A: actions, topics

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
        self._num_users = max_num_users
        self._items = {}
        self._num_items = max_num_items
        self._rated_items = collections.defaultdict(set)
        self._topics = {}
        self._num_topics = max_num_topics

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor  
        self.epsilon = epsilon

        self.QTable = np.empty([self._num_users, self._num_topics])
        self.times_rated = np.empty([self._num_users, self._num_topics])


    def reset(self, users=None, items=None, ratings=None):  
        """ Reset the recommender with optional starting user, item, and rating data.

        Parameters
        ----------
        users : dict, optional
            All starting users where the key is the user id while the value is the
            user features.
        items : dict, optional
            All starting items where the key is the user id while the value is the
            item features, in this case topic
        ratings : np.ndarray, optional
            All starting ratings where the key is a double whose first index is the
            id of the user making the rating and the second index is the id of the item being
            rated. The value is a double whose first index is the rating value and the second
            index is a numpy array that represents the context in which the rating was made.
        """        
        if users:
            self._users.update(users)
            self._num_users = len(self._users)
        if items:
            self._items.update(items)
            self._num_items = len(self._items)
            self._topics = dictinvert(items)
            self._num_topics = len(self._topics)

        self._rated_items = collections.defaultdict(set)

        # initialize Q table from initial ratings    
        if ratings:
            self.QTable = np.random.uniform(0, 5, self._num_users*self._num_topics).reshape([self._num_users,self._num_topics])
            self.times_rated = np.empty([self._num_users, self._num_topics])
            for (user_id, item_id), (rating, rating_context) in ratings.items():
                assert user_id in self._users
                assert item_id in self._items
                self._rated_items[user_id].add(item_id)
                item_topic = self._items[item_id]
                self.times_rated[user_id, item_topic] += 1
                # keep running average of the topic rating for this user
                self.QTable[user_id, item_topic] = rating*(1/self.times_rated[user_id, item_topic]) + 
                    self.QTable[user_id, item_topic]*(1-1/self.times_rated[user_id, item_topic])


    def update(self, users=None, items=None, ratings=None):
        """Update the recommender with new user, item, and rating data.
        
        Parameters
        ----------
        users : dict, optional
            All starting users where the key is the user id while the value is the
            user features.
        items : dict, optional
            All starting items where the key is the user id while the value is the
            item features, in this case topic
        ratings : np.ndarray, optional
            All starting ratings where the key is a double whose first index is the
            id of the user making the rating and the second index is the id of the item being
            rated. The value is a double whose first index is the rating value and the second
            index is a numpy array that represents the context in which the rating was made.
        """      
        if users:
            self._users.update(users)
            raise NotImplementedError()
        if items:
            raise NotImplementedError()
   
        if ratings:
            for (user_id, item_id), (rating, rating_context) in ratings.items():
                assert user_id in self._users
                assert item_id in self._items
                self._rated_items[user_id].add(item_id)
                item_topic = self._items[item_id]
                self.times_rated[user_id, item_topic] += 1
                self.QTable[user_id, item_topic] = (1-self.learning_rate)*self.QTable[user_id, item_topic] +
                    self.learning_rate*(rating + self.discount_factor*max(self.QTable))        

    def recommend(self, user_ids, num_recommendations = 1):
        if num_recommendations != 1:
            raise NotImplementedError()
        
        recs = np.zeros((len(user_ids), num_recommendations), dtype=np.int)
        for user_id in user_ids:
            prob = np.random.uniform
            if prob > self.epsilon:
                topic = np.argmax(self.QTable)
            else:
                topic = np.random.randint(0, self._num_topics) 

            # choose a random item of that category that user hasn't seen
            unseen_items = set(self._topics[topic]).difference(self._rated_items[user_id])
            if not unseen_items:
                raise NotImplementedError()
            recs[user_id] = np.random.choice(list(unseen_items))       

        return recs
  

    def dictinvert(d):
    inv = {}
    for k, v in d.iteritems():
        keys = inv.setdefault(v, [])
        keys.append(k)
    return inv