import torch
import torch.nn as nn
import torch.nn.functional as F



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
        output, hid = self.rnn(embeded)
        return output, hid


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, attention) -> None:
        super().__init__()
        self.attention = attention
        self.output_dim = output_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.RNN(emb_dim+hid_dim, hid_dim)
        self.linear = nn.Linear(hid_dim, output_dim)
    
    def forward(self, word_input: torch.LongTensor, hid, encoder_output):
        encoder_output = encoder_output.permute(1, 0, 2)
        values = self.attention(encoder_output, hid)
        values = values.unsqueeze(1)
        weights = torch.bmm(values, encoder_output).permute(1, 0, 2)
        word_input = word_input.unsqueeze(0)
        embeded =  self.embedding(word_input)
        embeded = torch.cat((embeded, weights), dim=-1).permute(1, 0, 2)
        output, hid =  self.rnn(embeded, hid)
        output = self.linear(output)
        return output, hid
    

class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, device) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        assert(encoder.hid_dim == decoder.hid_dim)

    def forward(self, src, trg):
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        outputs = torch.zeros(trg_len, batch_size, self.decoder.output_dim).to(self.device)
        encoder_output, hid = self.encoder(src)
        input = trg[:, 0]
        for i in range(1, trg.shape[1]):
            output, hid = self.decoder(input, hid, encoder_output)
            outputs[i] = output
            input = trg[:, i]   
        return outputs
    
class Attention(nn.Module):
    def __init__(self, enc_emb_dim, dec_hid_dim) -> None:
        super().__init__()
        self.attn = nn.Linear(enc_emb_dim+dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)
        
    def forward(self, output, hid):
        hid = hid.permute(1, 0, 2)
        hid = hid.repeat(1, output.shape[1], 1)
        kq =  torch.tanh(self.attn(torch.cat((hid, output), dim=-1)))
        attention = self.v(kq).squeeze(-1)
        return F.softmax(attention, dim=-1)

        
        