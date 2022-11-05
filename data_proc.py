import torch
from utils import normalize_string
from torch.utils.data import Dataset


class Lang(Dataset):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {0: '<sos>', 1: '<eos>'}
        self.n_words = 2
    
    def append(self, chs):
        for ch in chs:
            if ch not in self.word2idx.keys():
                self.word2idx[ch] = len(self.word2idx)
                self.idx2word[len(self.idx2word)] = ch
                self.n_words += 1
                
    def idxs2words(self, idxs):
        return [0] + [self.idx2word[idx] for idx in idxs] + [1]
    
    def words2idxs(self, words):
        return [0] + [self.word2idx[word] for word in words] + [1]

class ML_Dataset(Dataset):
    def __init__(self, data_path) -> None:
        super().__init__()
        self.data_path = data_path
        f = open(self.data_path, 'r')
        lines = f.readlines()
        self.pairs = [[normalize_string(sent) for sent in line.strip('\n').strip().split('\t')] for line in lines]
        self.CN_data = Lang()
        self.EN_data = Lang()
        for pair in self.pairs:
            self.CN_data.append(pair[0])
            self.EN_data.append(pair[1].split())
            
          
    def __getitem__(self, index):
        pairs = self.pairs[index]
        return torch.LongTensor(self.CN_data.words2idxs(pairs[0])), torch.LongTensor(self.EN_data.words2idxs(pairs[1].split()))
    
    def __len__(self):
        return len(self.pairs)
        

        
        