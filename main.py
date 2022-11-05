import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_proc import ML_Dataset
from models import Encoder, Decoder, Seq2Seq
import torch.optim as optim


DATA_PATH = 'cn-eng.txt'
train_dataset = ML_Dataset(data_path=DATA_PATH)
ENC_OUTPUT_DIM = train_dataset.CN_data.n_words
DEC_OUTPUT_DIM = train_dataset.EN_data.n_words
DEC_HID_DIM = ENC_HID_DIM = 128
DEC_EMB_DIM = ENC_EMB_DIM = 128
device = 'cuda' if torch.cuda.is_available() else 'cpu'
lr = 0.001

encoder = Encoder(ENC_OUTPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM)
decoder = Decoder(DEC_OUTPUT_DIM, DEC_EMB_DIM, DEC_HID_DIM)
seq2seq = Seq2Seq(encoder=encoder, decoder=decoder)

seq2seq.to(device)

dataloader = DataLoader(train_dataset, shuffle=True, batch_size=1)

optimizer = optim.Adam(seq2seq.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

for idx, (ch_sent, en_sent) in enumerate(dataloader):
    outputs = seq2seq(ch_sent, en_sent)
    outputs = outputs.view(-1, decoder.output_dim)
    en_sent = en_sent.view(-1)
    loss = criterion(outputs, en_sent)
    loss.backward()
    optimizer.zero_grad()
    optimizer.step()
    