from email.policy import default
import torch
import torch.nn as nn
from torch.optim import Adam
import argparse
import numpy as np
from zad1 import *

lr = 1e-4
batch_size_train = 10
batch_size_test = 32

def calculate_metrics(y_pred, y_true):
    tp = torch.sum(y_pred[y_true == 1] == 1)
    fp = torch.sum(y_pred[y_true == 0] == 1)
    tn = torch.sum(y_pred[y_true == 0] == 0)
    fn = torch.sum(y_pred[y_true == 1] == 0)
    return tp.item(), fp.item(), tn.item(), fn.item(), np.array([[tp.item(), fp.item()], [fn.item(), tn.item()]])

def print_metrics(acc, f1):
    print(f' Accuracy - {acc}')
    print(f'       F1 - {f1}')

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
    p = (tp_)/(tp_ + fp_)
    r = (tp_)/(tp_ + fn_)
    print_metrics(
        (tp_ + tn_)/(tp_ + tn_ + fp_ + fn_),
        (2*p*r)/(p+r)
    )
    print()

def evaluate(model, data, criterion, args):
    model.eval()
    loss_ = []
    tp_, fp_, tn_, fn_ = 0, 0, 0, 0
    cm_ = np.array([[0, 0], [0, 0]])
    with torch.no_grad():
        for batch_num, batch in enumerate(data):
            x, y, l = batch
            # x = x.cuda()
            # y = y.cuda()
            logits = model(x)
            y_pred = (torch.sigmoid(logits) > 0.5).int()
            loss = criterion(logits, y.float())
        
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

def load_dataset(max_size=-1, min_freq=0):
    train_dataset = NLPDataset()
    train_dataset.from_file('./data/sst_train_raw.csv')
    train_dataset.build_vocabs(max_size, min_freq)

    valid_dataset = NLPDataset()
    valid_dataset.from_file('./data/sst_valid_raw.csv')
    valid_dataset.example_vocab = train_dataset.example_vocab
    valid_dataset.label_vocab = train_dataset.label_vocab
    
    test_dataset = NLPDataset()
    test_dataset.from_file('./data/sst_test_raw.csv')
    test_dataset.example_vocab = train_dataset.example_vocab
    test_dataset.label_vocab = train_dataset.label_vocab

    return train_dataset, valid_dataset, test_dataset

class Baseline(nn.Module):
    def __init__(self, args, embeddings):
        super(Baseline, self).__init__()
        self.l1 = nn.Linear(300, 150)
        self.l2 = nn.Linear(150, 150)
        self.l3 = nn.Linear(150, 1)
        self.embeddings = embeddings
        
    def forward(self, x):
        x = self.embeddings(x)

        # non_empty_mask = x.abs().sum(dim=1).bool()
        # x = x[non_empty_mask]

        x = torch.mean(x, dim=1)
        # x = torch.amax(x, dim=1)
        
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        x = self.l3(x)
        return x.reshape(-1)

def initialize_model(args, embeddings): 
    return Baseline(args, embeddings)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=5)
    args = parser.parse_args()
    
    print('Loading data . . .')
    train_dataset, valid_dataset, test_dataset = load_dataset()
    print('Model initialization . . .')
    # embeddings = train_dataset.example_vocab.get_embedding_matrix_glove('./data/sst_glove_6b_300d.txt')
    embeddings = train_dataset.example_vocab.get_embedding_matrix_random()
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
# main()

# 1) 
#      Loss - 0.5463786928781441
#  Accuracy - 0.7385321100917431
#        F1 - 0.7719999999999999


# 2)
#      Loss - 0.47325233795813154
#  Accuracy - 0.7775229357798165
#        F1 - 0.7805429864253394


# 3)
#      Loss - 0.5585094376334122
#  Accuracy - 0.7327981651376146
#        F1 - 0.7615148413510747

# 4)
#      Loss - 0.6475826986134052
#  Accuracy - 0.698394495412844
#        F1 - 0.6103703703703705

# 5)
#      Loss - 0.47200672434909
#  Accuracy - 0.7775229357798165
#        F1 - 0.781038374717833
