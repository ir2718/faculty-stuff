import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from zad2 import load_dataset, evaluate, calculate_metrics, print_metrics, pad_collate_fn
import argparse

lr = 1e-4
batch_size_train = 10
batch_size_test = 32

class Model(nn.Module):
    def __init__(self, args, embeddings, cell, cell_args):
        super(Model, self).__init__()
        self.rnn1 = cell(cell_args[0], cell_args[1], num_layers=cell_args[2])
        self.l1 = nn.Linear(cell_args[1], 150)
        self.l2 = nn.Linear(150, 1)
        self.num_layers=cell_args[2]
        self.hidden_dim = cell_args[1]
        self.embeddings = embeddings

    def forward(self, x):
        x = self.embeddings(x)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)
        x = torch.transpose(x, 0, 1)
        x, h1 = self.rnn1(x, h0)
        x = x[-1]
        x = torch.relu(self.l1(x))
        x = self.l2(x)
        return x.reshape(-1)

def initialize_model(args, embeddings, cell, cell_args):
    model = Model(args, embeddings, cell, cell_args)
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
    parser.add_argument('--clip', default=0.8)
    args = parser.parse_args()


    print('Loading data . . .')
    train_dataset, valid_dataset, test_dataset = load_dataset()
    print('Model initialization . . .')
    embeddings = train_dataset.example_vocab.get_embedding_matrix_glove('./data/sst_glove_6b_300d.txt')
    # embeddings = train_dataset.example_vocab.get_embedding_matrix_random()

    cell = nn.GRU
    cell_args = (300, 150, 2)
    model = initialize_model(args, embeddings, cell, cell_args)

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

####### GRU REZULTATI ##########
#
# 1)
#    Loss - 0.5071575418114662
#  Accuracy - 0.7775229357798165
#        F1 - 0.7610837438423644

# 2)
#      Loss - 0.4430552933897291
#  Accuracy - 0.8084862385321101
#        F1 - 0.8073817762399077

####### LSTM REZULTATI ##########
#
# 1)
#      Loss - 0.46019559087497847
#  Accuracy - 0.7901376146788991
#        F1 - 0.7941507311586051

# 2)
#      Loss - 0.45048272077526363
#  Accuracy - 0.7912844036697247
#        F1 - 0.7898383371824481




####### LSTM REZULTATI, NUM_LAYERS = 1 ##########
#      Loss - 0.6339511552027294
#  Accuracy - 0.7178899082568807
#        F1 - 0.7588235294117647

####### LSTM REZULTATI, NUM_LAYERS = 3 ##########
#      Loss - 0.776613076882703
#  Accuracy - 0.6915137614678899
#        F1 - 0.6014814814814815

####### LSTM REZULTATI, HIDDEN_SIZE = 20 ##########
#      Loss - 0.49468992224761416
#  Accuracy - 0.7603211009174312
#        F1 - 0.7544065804935371

####### LSTM REZULTATI, HIDDEN_SIZE = 300 ##########
#      Loss - 0.5075443227376256
#  Accuracy - 0.7947247706422018
#        F1 - 0.7742749054224465




####### GRU REZULTATI, RANDOM EMBEDDING ##########
#     Loss - 0.6425309889018536
# Accuracy - 0.7213302752293578
#       F1 - 0.6951066499372647

####### BASELINE REZULTATI, RANDOM EMBEDDING ##########
#     Loss - 0.5568012669682503
# Accuracy - 0.7419724770642202
#       F1 - 0.7392815758980301




####### GRU REZULTATI, MIN FREQ 20 ##########
#     Loss - 0.5528865775891713
# Accuracy - 0.7305045871559633
#       F1 - 0.7638190954773869
####### GRU REZULTATI, MIN FREQ 80 ##########
#     Loss - 0.6082264717136111
# Accuracy - 0.6777522935779816
#       F1 - 0.7088082901554406


####### GRU REZULTATI, BATCH SIZE 64 ##########
#      Loss - 0.4708492181130818
#  Accuracy - 0.7809633027522935
#        F1 - 0.7865921787709498
####### GRU REZULTATI, BATCH SIZE 128 ##########
#      Loss - 0.571973443031311
#  Accuracy - 0.7144495412844036
#        F1 - 0.7502507522567703


####### GRU REZULTATI, SGD ##########
#      Loss - 0.6931191576378686
#  Accuracy - 0.4908256880733945
#        F1 - 0.6584615384615384
####### GRU REZULTATI, RMSPROP ##########
#      Loss - 0.5249063936727387
#  Accuracy - 0.7878440366972477
#        F1 - 0.758800521512386


####### GRU REZULTATI, CLIP 0.1 ##########
#      Loss - 0.5747147904975074
#  Accuracy - 0.7465596330275229
#        F1 - 0.69432918395574
####### GRU REZULTATI, CLIP 0.8  ##########
#      Loss - 0.4440167854939188
#  Accuracy - 0.7821100917431193
#        F1 - 0.7952586206896551



####### GRU REZULTATI, LEAKY RELU ##########
#      Loss - 0.4631889451827322
#  Accuracy - 0.7912844036697247
#        F1 - 0.7690355329949239
####### GRU REZULTATI, SIGMOIDA ##########
#      Loss - 0.5241121239960194
#  Accuracy - 0.7626146788990825
#        F1 - 0.7329032258064515