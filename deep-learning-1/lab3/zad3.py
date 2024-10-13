import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from zad2 import load_dataset, evaluate, calculate_metrics, print_metrics, pad_collate_fn
import argparse

lr = 1e-4
batch_size_train = 10
batch_size_test = 32

class RNN1(nn.Module):
    def __init__(self, args, embeddings):
        super(RNN1, self).__init__()
        self.rnn1 = nn.RNN(300, 150, num_layers=2)
        self.l1 = nn.Linear(150, 150)
        self.l2 = nn.Linear(150, 1)
        self.embeddings = embeddings
        
    def forward(self, x):
        x = self.embeddings(x)
        h0 = torch.zeros(2, x.size(0), 150)
        x = torch.transpose(x, 0, 1)
        x, h1 = self.rnn1(x, h0)
        x = x[-1]
        x = torch.relu(self.l1(x))
        x = self.l2(x)
        return x.reshape(-1)

def initialize_model(args, embeddings):
    model = RNN1(args, embeddings)
    return model

def train(model, data, optimizer, criterion, args):
    model.train()
    loss_ = []
    tp_, fp_, tn_, fn_ = 0, 0, 0, 0
    cm_ = np.array([[0, 0], [0, 0]])
    for batch_num, batch in enumerate(data):
        model.zero_grad()
        x, y, l = batch
        # x = x.cuda()
        # y = y.cuda()
        logits = model(x)
        y_pred = (torch.sigmoid(logits.detach()) > 0.5).int()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        loss = criterion(logits, y.float())
        loss.backward()
        optimizer.step()
        
        tp, fp, tn, fn, cm = calculate_metrics(y_pred.detach(), y.detach())
        tp_ += tp
        fp_ += fp
        tn_ += tn
        fn_ += fn
        cm_ += cm
        loss_.append(loss.item())

    print(f'Loss - {np.sum(loss_)/len(loss_)}')
    p = 0 if (tp_ + fp_) == 0 else (tp_)/(tp_ + fp_)
    r = 0 if (tp_ + fn_) == 0 else (tp_)/(tp_ + fn_)
    print_metrics(
        (tp_ + tn_)/(tp_ + tn_ + fp_ + fn_),
        0 if (p+r) == 0 else (2*p*r)/(p+r)
    )
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=5)
    parser.add_argument('--clip', default=0.25)
    args = parser.parse_args()


    print('Loading data . . .')
    train_dataset, valid_dataset, test_dataset = load_dataset()
    print('Model initialization . . .')
    embeddings = train_dataset.example_vocab.get_embedding_matrix_glove('./data/sst_glove_6b_300d.txt')
    model = initialize_model(args, embeddings)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    print('Setting up dataloaders . . .')
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size_train, shuffle=True, collate_fn=pad_collate_fn)    
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=batch_size_test, shuffle=False, collate_fn=pad_collate_fn)   
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size_test, shuffle=False, collate_fn=pad_collate_fn)

    for e in range(args.epochs):
        print(f' ---------------------------------- Epoch {e} ---------------------------------- ')
        print(f' --- Training ---')
        train(model, train_dataloader, optimizer, criterion, args)
        print(f'\n --- Validation ---')
        evaluate(model, valid_dataloader, criterion, args)
        print()
    
    print(f'\n --- Testing ---')
    evaluate(model, test_dataloader, criterion, args)

main()

# 1)
#      Loss - 0.5955881401896477
#  Accuracy - 0.7144495412844036
#        F1 - 0.6675567423230975

# 2)
#     Loss - 0.5009605778115136
# Accuracy - 0.7626146788990825
#       F1 - 0.7898477157360406

# 3)
#      Loss - 0.5092725647347314
#  Accuracy - 0.7637614678899083
#        F1 - 0.7780172413793104

# 4)
#      Loss - 0.5885856917926243
#  Accuracy - 0.75
#        F1 - 0.7295285359801489

# 5)
#      Loss - 0.5132158994674683
#  Accuracy - 0.7694954128440367
#        F1 - 0.7827027027027028