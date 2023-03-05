import torch
import torch.nn as nn

class GCN_Classifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,
                 dropout, return_embeds=False):

        super(GCN_Classifier, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        # Define the batch normalization layers
        self.bn1 = nn.BatchNorm1d(num_features=hidden_dim) 
        self.bn2 = nn.BatchNorm1d(num_features=hidden_dim)

        self.softmax = nn.LogSoftmax()
        self.dropout = nn.Dropout(p=dropout)
        self.ReLU = nn.ReLU()

        # Skip classification layer and return node embeddings
        self.return_embeds = return_embeds

    def forward(self, x, transition_matrix):

        z1 = torch.sparse.mm(transition_matrix, x)
        z1 = self.fc1(z1)
        z1 = self.bn1(z1)
        z1 = self.ReLU(z1)
        z1 = self.dropout(z1)
        
        z2 = torch.sparse.mm(transition_matrix, z1)
        z2 = self.fc2(z2)
        z2 = self.bn2(z2)
        z2 = self.ReLU(z2)
        z2 = self.dropout(z2)

        z3 = torch.sparse.mm(transition_matrix, z2)
        z3 = self.fc3(z3)

        if self.return_embeds:
            return z3
        else:
            return self.softmax(z3)
