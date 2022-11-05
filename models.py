import torch
import torch.nn as nn



class Encoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.RNN(emb_dim, hid_dim)
    
    def forward(self, words_input:torch.LongTensor):
        words_input, words_input.unsqueeze(0)
        embeded = self.embedding(words_input)
        embeded = embeded.permute(1, 0, 2)
        _, hid = self.rnn(embeded)
        return hid


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.RNN(emb_dim, hid_dim)
        self.linear = nn.Linear(hid_dim, output_dim)
    
    def forward(self, word_input: torch.LongTensor, hid):
        word_input = word_input.unsqueeze(0)
        embeded =  self.embedding(word_input)
        embeded = embeded.permute(1, 0, 2)
        output, hid =  self.rnn(embeded, hid)
        output = self.linear(output)
        return output, hid
    

class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        assert(encoder.hid_dim == decoder.hid_dim)

    def forward(self, src, trg):
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        outputs = torch.zeros(trg_len, batch_size, self.decoder.output_dim)
        hid = self.encoder(src)
        input = trg[:, 0]
        for i in range(1, trg.shape[1]):
            output, hid = self.decoder(input, hid)
            outputs[i] = output
            input = trg[:, i]   
        return outputs
        
        