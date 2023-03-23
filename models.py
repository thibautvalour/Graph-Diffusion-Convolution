import torch
import torch.nn as nn

class GCN_Classifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim,
                 hidden_layers, output_dim,
                 dropout, return_embeds=False):

        super(GCN_Classifier, self).__init__()

        self.hidden_layers = hidden_layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(num_features=hidden_dim)
        self.clf = nn.Linear(hidden_dim, output_dim)

        self.softmax = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(p=dropout)
        self.ReLU = nn.ReLU()

        # Skip classification layer and return node embeddings
        self.return_embeds = return_embeds

    def forward(self, x, transition_matrix):

        z = torch.sparse.mm(transition_matrix, x)
        z = self.fc1(z)
        z = self.bn1(z)
        z = self.ReLU(z)
        z = self.dropout(z)
        
        for _ in range(self.hidden_layers-1):
            z = torch.sparse.mm(transition_matrix, z)
            z = nn.Linear(self.hidden_dim, self.hidden_dim)(z)
            z = nn.BatchNorm1d(num_features=self.hidden_dim)(z)
            z = self.ReLU(z)
            z = self.dropout(z)

        z = torch.sparse.mm(transition_matrix, z)
        z = self.clf(z)

        if self.return_embeds:
            return z
        else:
            return self.softmax(z)
