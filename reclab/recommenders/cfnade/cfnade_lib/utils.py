""" Util functions for class Cfnade"""
from itertools import islice
import numpy as np
from keras import backend as K


def data_gen(
         ratings_df, batch_size,
         num_users, mode):
    """
    a generator function yields inputs for each batch

    ratings_df: rating matrix, num_iters * num_users, entry is input rating rounded to integer
    batch_size: int, batch size, default is 64
    num_users: int, number of users, user_id starts from 0
    mode: int, 0 indicates train_set, 2 indicates test_set
    """
    while True:
        next_n_data_lines = np.asarray(list(islice(ratings_df, batch_size)))
        if not next_n_data_lines:
            break

        input_ranking_vectors = np.zeros((batch_size, num_users, 5), dtype='int8')
        output_ranking_vectors = np.zeros((batch_size, num_users, 5), dtype='int8')
        input_mask_vectors = np.zeros((batch_size, num_users), dtype='int8')
        output_mask_vectors = np.zeros((batch_size, num_users), dtype='int8')
        for i, line in enumerate(next_n_data_lines):
            if mode == 0:
                # a random ordered list 0 to len(user_ids)-1
                ordering = np.random.permutation(np.arange(num_users))
                random_num = np.random.randint(0, len(ordering))
                flag_in = (ordering < random_num)
                flag_out = (ordering >= random_num)
                user_ids = range(num_users)
                input_mask_vectors[i][users_ids] = flag_in
                output_mask_vectors[i][users_ids] = flag_out

                for j, (user_id, value) in enumerate(zip(user_ids, line)):
                    if flag_in[j]:
                        input_ranking_vectors[i, user_id, (value-1)] = 1
                    else:
                        output_ranking_vectors[i, user_id, (value-1)] = 1

        inputs = {
            'input_ratings': input_ranking_vectors,
            'output_ratings': output_ranking_vectors,
            'input_masks': input_mask_vectors,
            'output_masks': output_mask_vectors}

        outputs = {'nade_loss': np.zeros([batch_size])}
        yield (inputs, outputs)


def prediction_layer(x):
    # x.shape = (?,6040,5)
    x_cumsum = K.cumsum(x, axis=2)
    # x_cumsum.shape = (?,6040,5)

    output = K.softmax(x_cumsum)
    # output = (?,6040,5)
    return output


def prediction_output_shape(input_shape):

    return input_shape


def d_layer(x):

    return K.sum(x, axis=1)


def d_output_shape(input_shape):

    return (input_shape[0], )


def D_layer(x):

    return K.sum(x, axis=1)


def D_output_shape(input_shape):

    return (input_shape[0],)


def rating_cost_lambda_func(args):
    alpha = 1.
    std = 0.01
    pred_score, true_ratings, input_masks, output_masks, D, d = args
    pred_score_cum = K.cumsum(pred_score, axis=2)
    prob_item_ratings = K.softmax(pred_score_cum)
    accu_prob_1N = K.cumsum(prob_item_ratings, axis=2)
    accu_prob_N1 = K.cumsum(prob_item_ratings[:, :, ::-1], axis=2)[:, :, ::-1]
    mask1N = K.cumsum(true_ratings[:, :, ::-1], axis=2)[:, :, ::-1]
    maskN1 = K.cumsum(true_ratings, axis=2)
    cost_ordinal_1N = -K.sum((K.log(prob_item_ratings) - K.log(accu_prob_1N)) * mask1N, axis=2)
    cost_ordinal_N1 = -K.sum((K.log(prob_item_ratings) - K.log(accu_prob_N1)) * maskN1, axis=2)
    cost_ordinal = cost_ordinal_1N + cost_ordinal_N1
    nll_item_ratings = K.sum(-(true_ratings * K.log(prob_item_ratings)), axis=2)
    nll = std * K.sum(nll_item_ratings, axis=1) * 1.0 * D / (D - d + 1e-6) \
        + alpha * K.sum(cost_ordinal, axis=1) * 1.0 * D / (D - d + 1e-6)
    cost = K.mean(nll)
    cost = K.expand_dims(cost, 0)

    return cost


class RMSE_eval(Callback):
    def __init__(self, data_set, training_set):

        self.data_set = data_set
        self.rmses = []
        self.rate_score = np.array([1, 2, 3, 4, 5], np.float32)
        self.training_set = training_set

    def eval_rmse(self):
        squared_error = []
        num_samples = []
        for i, batch in enumerate(self.data_set.generate(max_iters=1)):
            inp_r = batch[0]['input_ratings']
            out_r = batch[0]['output_ratings']
            inp_m = batch[0]['input_masks']
            out_m = batch[0]['output_masks']
            pred_batch = self.model.predict(batch[0])[1]
            true_r = out_r.argmax(axis=2) + 1
            pred_r = (pred_batch * self.rate_score[np.newaxis, np.newaxis, :]).sum(axis=2)
            mask = out_r.sum(axis=2)
            '''
            if i == 0:
                print [true_r[0][j] for j in np.nonzero(true_r[0]* mask[0])[0]]
                print [pred_r[0][j] for j in np.nonzero(pred_r[0]* mask[0])[0]]
            '''

            se = np.sum(np.square(true_r - pred_r) * mask)
            n = np.sum(mask)
            squared_error.append(se)
            num_samples.append(n)

        total_squared_error = np.array(squared_error).sum()
        total_num_samples = np.array(num_samples).sum()
        rmse = np.sqrt(total_squared_error / (total_num_samples * 1.0 + 1e-8))

        return rmse

    def on_epoch_end(self, epoch, logs={}):
        score = self.eval_rmse()
        if self.training_set:
            print('training set RMSE for epoch %d is %f ' % (epoch, score))
        else:
            print('validation set RMSE for epoch %d is %f ' % (epoch, score))

        self.rmses.append(score)
