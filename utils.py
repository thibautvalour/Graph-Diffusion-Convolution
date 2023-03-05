import torch

def train(model, data, train_idx, adj_t, optimizer, loss_fn):
    model.train()

    optimizer.zero_grad()
    out = model(data.x, adj_t)
    loss = loss_fn(out[train_idx], data.y[train_idx].reshape(-1))
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def test(model, data, split_idx, adj_t, evaluator):
    model.eval()

    # The output of model on all data
    out = model(data.x, adj_t)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc
