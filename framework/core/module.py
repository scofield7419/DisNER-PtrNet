import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from framework.utils.common import *


class WordEmbeddings(nn.Module):
    def __init__(self, vocab_size, embed_dim, pre_trained_embed_matrix, drop_out_rate):
        super(WordEmbeddings, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embeddings.weight.data.copy_(torch.from_numpy(pre_trained_embed_matrix))
        self.dropout = nn.Dropout(drop_out_rate)

    def forward(self, words_seq):
        word_embeds = self.embeddings(words_seq)
        word_embeds = self.dropout(word_embeds)
        return word_embeds

    def weight(self):
        return self.embeddings.weight


class CharEmbeddings(nn.Module):
    def __init__(self, vocab_size, embed_dim, drop_out_rate):
        super(CharEmbeddings, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(drop_out_rate)

    def forward(self, words_seq):
        char_embeds = self.embeddings(words_seq)
        char_embeds = self.dropout(char_embeds)
        return char_embeds


class Encoder(nn.Module):

    def __init__(self, input_dim, hidden_dim,
                 word_vocab_len, char_vocab_len, word_embed_matrix,
                 layers, is_bidirectional, drop_out_rate):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.is_bidirectional = is_bidirectional
        self.drop_rate = drop_out_rate
        self.word_embeddings = WordEmbeddings(word_vocab_len, word_embed_dim, word_embed_matrix, drop_rate)
        self.char_embeddings = CharEmbeddings(char_vocab_len, char_embed_dim, drop_rate)
        self.pos_embeddings = nn.Embedding(max_positional_idx, positional_embed_dim, padding_idx=0)
        if enc_type == 'LSTM':
            self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.layers, batch_first=True,
                                bidirectional=self.is_bidirectional)
        self.dropout = nn.Dropout(self.drop_rate)
        self.conv1d = nn.Conv1d(char_embed_dim, char_feature_size, conv_filter_size)
        self.max_pool = nn.MaxPool1d(max_word_len + conv_filter_size - 1, max_word_len + conv_filter_size - 1)


    def forward(self, words, chars, pos_seq, is_training=False):
        src_word_embeds = self.word_embeddings(words)
        pos_embeds = self.dropout(self.pos_embeddings(pos_seq))
        char_embeds = self.char_embeddings(chars)
        char_embeds = char_embeds.permute(0, 2, 1)

        char_feature = torch.tanh(self.max_pool(self.conv1d(char_embeds)))
        char_feature = char_feature.permute(0, 2, 1)

        words_input = torch.cat((src_word_embeds, char_feature, pos_embeds), -1)
        if enc_type == 'LSTM':
            outputs, hc = self.lstm(words_input)

        outputs = self.dropout(outputs)
        return outputs, words_input


class Pointer(nn.Module):
    def __init__(self, input_dim):
        super(Pointer, self).__init__()
        self.input_dim = input_dim
        self.linear_info = nn.Linear(dec_hidden_size, self.input_dim, bias=False)
        self.linear_ctx = nn.Linear(self.input_dim, self.input_dim, bias=False)
        self.linear_query = nn.Linear(self.input_dim, self.input_dim, bias=True)
        self.projection = nn.Sequential(
            nn.Linear(3 * self.input_dim, 250),
            nn.Dropout(drop_rate),
            nn.ReLU(True),
            nn.Linear(250, 1)
        )

    def forward(self, s_prev, enc_hs, cur_men_rep, src_mask):
        src_time_steps = enc_hs.size()[1]

        dh = self.linear_info(cur_men_rep)
        uh = self.linear_ctx(enc_hs)
        wq = self.linear_query(s_prev)
        pointer_input = torch.cat((uh, wq.repeat(1, src_time_steps, 1), dh.repeat(1, src_time_steps, 1)), 2)
        wquh = torch.tanh(pointer_input)
        attn_weights = self.projection(wquh).squeeze()
        attn_weights.data.masked_fill_(src_mask.squeeze().byte().data, -float('inf'))
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights_ = attn_weights.unsqueeze(0).unsqueeze(-1)
        ctx = (enc_hs * attn_weights_).sum(dim=1)
        return attn_weights, ctx


class SelfAttention1(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention1, self).__init__()
        self.input_dim = input_dim
        self.projection = nn.Sequential(
            nn.Linear(self.input_dim, 50),
            nn.Dropout(drop_rate),
            nn.ReLU(True),
            nn.Linear(50, 1)
        )

    def forward(self, hid_rep):
        hid_rep = torch.stack(hid_rep)
        hid_rep = hid_rep.unsqueeze(0)
        energy = self.projection(hid_rep)
        weights = F.softmax(energy.squeeze(-1), dim=1)
        att_rep = (hid_rep * weights.unsqueeze(-1)).sum(dim=1)

        return att_rep



class SelfAttention2(nn.Module):
    def __init__(self, input_dim, drop_out_rate):
        super(SelfAttention2, self).__init__()
        self.input_dim = input_dim
        self.drop_rate = drop_out_rate
        self.pointer_lstm = nn.LSTM(2 * self.input_dim, self.input_dim, 1, batch_first=True,
                                    bidirectional=True)

        self.pointer_lin = nn.Linear(2 * self.input_dim, 1)
        self.dropout = nn.Dropout(self.drop_rate)


    def forward(self, hid_rep):
        src_time_steps = hid_rep.size()[1]

        pointer_lstm_out, phc = self.pointer_lstm(hid_rep)
        pointer_lstm_out = self.dropout(pointer_lstm_out)
        ptr_ = self.pointer_lin(pointer_lstm_out).squeeze()
        ptr_weights = F.softmax(ptr_, dim=-1)
        att_rep = torch.bmm(ptr_weights, hid_rep  )  # .squeeze()

        return att_rep



class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers, drop_out_rate, max_length):
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.drop_rate = drop_out_rate
        self.max_length = max_length

        self.mPointer = Pointer(input_dim)
        self.lstm = nn.LSTMCell(self.input_dim + enc_hidden_size + enc_inp_size, self.hidden_dim)

        self.pointer_lstm = nn.LSTM(2 * self.input_dim, self.input_dim, 1, batch_first=True,
                                    bidirectional=True)

        self.pointer_lin = nn.Linear(2 * self.input_dim, 1)
        self.dropout = nn.Dropout(self.drop_rate)

        self.out_lin = nn.Linear(3 * self.input_dim, max_src_len)


    def forward(self, dec_inp, h_prev, enc_hs, cur_men_rep, src_mask, is_training=False):
        src_time_steps = enc_hs.size()[1]
        hidden, cell_state = self.lstm(dec_inp, h_prev)
        hidden = self.dropout(hidden)
        ptr_weights, ctx = self.mPointer(hidden, enc_hs, cur_men_rep, src_mask)
        out_input = torch.cat((ctx, hidden, cur_men_rep), -1)
        att_out_lin = self.out_lin(out_input)
        att_out_lin.data.masked_fill_(src_mask.squeeze().byte().data, -float('inf'))
        att_out_ = F.softmax(att_out_lin.squeeze(), dim=-1)

        return (hidden, cell_state), att_out_


