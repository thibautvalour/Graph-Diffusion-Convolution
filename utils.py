import torch
from sklearn.metrics import accuracy_score

def train(model, data, train_idx, adj_t, optimizer, loss_fn):
    model.train()

    optimizer.zero_grad()
    out = model(data.x, adj_t)
    loss = loss_fn(out[train_idx], data.y[train_idx].reshape(-1))
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def test(model, data, train_idx, test_idx, adj_t):
    model.eval()

    out = model(data.x, adj_t)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = accuracy_score(data.y[train_idx].cpu(), y_pred[train_idx].cpu())
    test_acc = accuracy_score(data.y[test_idx].cpu(), y_pred[test_idx].cpu())

    return train_acc, test_acc
