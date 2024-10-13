from torch.utils.data import Dataset, DataLoader
import torch

class NLPDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.instances = []
        self.example_vocab = None
        self.label_vocab = None
        self.example_freq = {}
        self.label_freq = {}

    def from_file(self, path):
        for line in open(path, 'r', encoding='utf-8'):
            example, label = line.strip().split(', ')
            label = label.strip()
            example = example.split()

            for w in example:
                if w not in self.example_freq.keys():
                    self.example_freq[w] = 0
            
            if label not in self.label_freq.keys():
                self.label_freq[label] = 0

            self.instances.append((example, label))

    def __getitem__(self, idx):
        return torch.tensor([self.example_vocab.encode(i) for i in self.instances[idx][0]]), \
               torch.tensor([self.label_vocab.encode(self.instances[idx][1])])

    def __len__(self):
        return len(self.instances)

    def get_frequencies_examples(self):
        vocab_list = []
        [vocab_list.extend(w) for w, _ in self.instances]

        for w in self.example_freq.keys():
            self.example_freq[w] = vocab_list.count(w)

        return self.example_freq

    def get_frequencies_labels(self):
        vocab_list = [y for _, y in self.instances]

        for y in self.label_freq.keys():
            self.label_freq[y] = vocab_list.count(y)
        
        return self.label_freq

    def build_vocabs(self, max_size=-1, min_freq=0):
        self.example_vocab = Vocab(self.get_frequencies_examples(), max_size, min_freq)
        self.label_vocab = Vocab(self.get_frequencies_labels(), max_size, min_freq, label=True)

class Vocab():
    def __init__(self, frequencies, max_size=-1, min_freq=0, label=False):
        freq = {k:v for k, v in frequencies.items() if v > min_freq}
        freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        freq = freq if max_size == -1 else freq[:max_size]


        self.label = label
        self.stoi, self.itos = {}, {}

        add = [('<PAD>', 0), ('<UNK>', 1)]
        if not label:
            for w, i in add:
                self.stoi[w] = i
                self.itos[i] = w 

        j = 0
        for w, _ in freq:
            ind = j + len(add) if not label else j
            self.stoi[w] = ind
            self.itos[ind] = w 
            j += 1

    def encode(self, s):
        if isinstance(s, list):
            return torch.tensor([self.encode(x).item() for x in s])
        
        # if self.label:
            # print(self.stoi)
            # print(f'returning { s, torch.tensor(self.stoi.get(s, self.stoi.get("<UNK>", 0))) }')
        return torch.tensor(self.stoi.get(s, self.stoi.get('<UNK>', 0)))
        
    def get_embedding_matrix_random(self):
        dims = 300
        embedding = torch.normal(mean=0, std=1, size=(len(self.stoi), dims))
        embedding[0] = torch.tensor(0) # padding
        return torch.nn.Embedding.from_pretrained(embedding, padding_idx=0, freeze=False)

    def get_embedding_matrix_glove(self, path):
        # koristiti torch.nn.Embedding.from_pretrained(padding_idx=0, freeze=True)
        dims = 300
        embedding = torch.normal(mean=0, std=1, size=(len(self.stoi), dims))
        embedding[0] = torch.tensor(0) # padding

        for line in open(path, 'r', encoding='utf-8'):
            arr = line.split()
            w, e = arr[0], [float(x) for x in arr[1:]]
            row = self.stoi.get(w, self.stoi.get('<UNK>'))
            embedding[row] = torch.tensor(e)

        return torch.nn.Embedding.from_pretrained(embedding, padding_idx=0, freeze=True)

def collate_fn(batch):
    """
    Arguments:
        Batch:
        list of Instances returned by `Dataset.__getitem__`.
    Returns:
        A tensor representing the input batch.
    """

    texts, labels = zip(*batch) # Assuming the instance is in tuple-like form
    labels = torch.cat(labels)
    lengths = torch.tensor([len(text) for text in texts]) # Needed for later
    # Process the text instances
    return texts, labels, lengths

def pad_collate_fn(batch, padding_value=0):
    #https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_sequence.html
    texts, labels, lengths = collate_fn(batch)
    return torch.nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=padding_value), labels, lengths

def test_method3(pad_index=0):
    batch_size = 2 # Only for demonstrative purposes
    shuffle = False # Only for demonstrative purposes
    train_dataset = NLPDataset()
    train_dataset.from_file('data/sst_train_raw.csv')
    train_dataset.build_vocabs()
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=pad_collate_fn)
    texts, labels, lengths = next(iter(train_dataloader))
    print(f"Texts: {texts}")
    print(f"Labels: {labels}")
    print(f"Lengths: {lengths}")

    # >>> Texts: tensor([[   2,  554,    7, 2872,    6,   22,    2, 2873, 1236,    8,   96, 4800,
    #                    4,   10,   72,    8,  242,    6,   75,    3, 3576,   56, 3577,   34,
    #                    2022, 2874, 7123, 3578, 7124,   42,  779, 7125,    0,    0],
    #                [   2, 2875, 2023, 4801,    5,    2, 3579,    5,    2, 2876, 4802,    7,
    #                    40,  829,   10,    3, 4803,    5,  627,   62,   27, 2877, 2024, 4804,
    #                    962,  715,    8, 7126,  555,    5, 7127, 4805,    8, 7128]])
    # >>> Labels: tensor([0, 0])
    # >>> Lengths: tensor([32, 34])


def test_method1(train_dataset, text_vocab, label_vocab):
    instance_text, instance_label = train_dataset.instances[3]
    print(f"Text: {instance_text}")                                         # Text: ['yet', 'the', 'act', 'is', 'still', 'charming', 'here']
    print(f"Label: {instance_label}")                                       # Label: positive

    print(f"Numericalized text: {text_vocab.encode(instance_text)}")        # Numericalized text: tensor([189,   2, 674,   7, 129, 348, 143])
    print(f"Numericalized label: {label_vocab.encode(instance_label)}")     # Numericalized label: tensor(0)
    

def test_method2(frequencies):
    print(frequencies['the']) # 5954
    print(frequencies['a']) # 4361
    print(frequencies['and']) # 3831
    print(frequencies['of']) # 3631

    text_vocab = Vocab(frequencies, max_size=-1, min_freq=0)
    print(text_vocab.stoi['<PAD>']) # 0
    print(text_vocab.stoi['<UNK>']) # 1
    print(text_vocab.stoi['the']) # 2
    print(text_vocab.stoi['a']) # 3
    print(text_vocab.stoi['and']) # 4
    print(text_vocab.stoi['of']) # 5

    print(text_vocab.itos[0]) # <PAD> 
    print(text_vocab.itos[1]) # <UNK>
    print(text_vocab.itos[2]) # the
    print(text_vocab.itos[3]) # a
    print(text_vocab.itos[4]) # and
    print(text_vocab.itos[5]) # of
    print(len(text_vocab.itos)) # 14806

    print(text_vocab.stoi['my']) # 188
    print(text_vocab.stoi['twists']) # 930
    print(text_vocab.stoi['lets']) # 956
    print(text_vocab.stoi['sports']) # 1275
    print(text_vocab.stoi['amateurishly']) # 6818


# --- test1 ----
# dataset = NLPDataset()
# dataset.from_file('./data/sst_train_raw.csv')
# dataset.build_vocabs()
# test_method1(dataset, dataset.example_vocab, dataset.label_vocab)

# --- test2 ----
# dataset = NLPDataset()
# dataset.from_file('./data/sst_train_raw.csv')
# freq = dataset.get_frequencies_examples()
# test_method2(freq)

# --- test3 ----
# dataset = NLPDataset()
# dataset.from_file('./data/sst_train_raw.csv')
# freq = dataset.get_frequencies_labels()
# print(freq)
# tmp = Vocab(freq, label=True)
# print(tmp.stoi, tmp.itos)

# --- test4 ---
# dataset = NLPDataset()
# dataset.from_file('./data/sst_train_raw.csv')
# dataset.build_vocabs()
# m = dataset.example_vocab.get_embedding_matrix_random()
# print(m.weight)
# m = dataset.example_vocab.get_embedding_matrix_glove(path='./data/sst_glove_6b_300d.txt')
# print(m.weight[143]) # 'here'


# --- test5 ---
# test_method3()