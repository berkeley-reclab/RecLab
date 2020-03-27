from recommenders.libfm import recommender
from recommenders.cf-nade import cf-nade_lib


class CFNade_Recommender(recommender.PredictRecommender):
	"""
    cf-nade recommender from the paper
	"A Neural Autoregressive Approach to Collaborative Filtering"

    The class supports the CF-Nade collaborative filtering.

    Parameters
    ----------
    
    """
	def _predict(self, user_item): 
		preds = []
        for user_id, item_id, _ in user_item:
        #add prediction for a user item pair using CF-Nade




        return np.array(preds)




