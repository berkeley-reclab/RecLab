import torch

class AutoRec(torch.nn.Module):
    def __init__(self, num_users, num_items,
                 seen_users, seen_items,
                 hidden_neuron,
                 dropout=0.05, random_seed=0):
        super(AutoRec, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.seen_users = seen_users
        self.seen_items = seen_items

        self.hidden_neuron = hidden_neuron
        self.random_seed = random_seed
        self.dropout_p = dropout
        self.sigmoid = torch.nn.Sigmoid()

    def _loss(self, pred, test, mask, lambda_value=1):
        mse = torch.nn.MSELoss()(pred * mask, test)
        reg_value_enc = torch.mul(lambda_value / 2, torch.square(list(self.encoder.parameters())[1].norm(p='fro')))
        reg_value_dec = torch.mul(lambda_value / 2, torch.square(list(self.decoder.parameters())[1].norm(p='fro')))
        return torch.add(mse, torch.add(reg_value_enc, reg_value_dec))

    def prepare_model(self):
        self.encoder = torch.nn.Linear(self.num_users, self.hidden_neuron, bias=True)
        self.dropout = torch.nn.Dropout(p=self.dropout_p)
        self.decoder = torch.nn.Linear(self.hidden_neuron, self.num_users, bias=True)

    def forward(self, x):
        x = self.encoder(x)
        x = self.sigmoid(x)
        x = self.dropout(x)
        x = self.decoder(x)
        return x

    def predict(self, user_item, test_data):
        users = [triple[0] for triple in user_item]
        items = [triple[1] for triple in user_item]

        user_item = zip(users, items)
        user_idx = set(users)
        item_idx = set(items)
        Estimated_R = self.forward(test_data)
        for user in range(test_data.shape[0]):
            for item in range(test_data.shape[1]):
                if user not in self.seen_users and item not in self.seen_items:
                    Estimated_R[user,item] = 3
        idx = [tuple(users), tuple(items)]
        Estimated_R = Estimated_R.clamp(1, 5)
        return Estimated_R.T[idx].cpu().detach().numpy()
