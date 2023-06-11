import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from framework.core.module import *
from framework.utils.common import *


class Seq2SeqModel(nn.Module):
    def __init__(self, word_vocab_len, char_vocab_len, word_embed_matrix):
        super(Seq2SeqModel, self).__init__()
        self.encoder = Encoder(enc_inp_size, int(enc_hidden_size / 2), word_vocab_len, char_vocab_len,
                               word_embed_matrix, 1, True, drop_rate)
        self.decoder = Decoder(dec_inp_size, dec_hidden_size, 1, drop_rate, max_trg_len)
        self.dropout = nn.Dropout(drop_rate)

        init_wt = np.ones_like(np.arange(max_src_len))
        init_wt[0] = 50
        weit = torch.FloatTensor(init_wt)
        if torch.cuda.is_available():
            weit = weit.cuda()
        self.myLoss = nn.NLLLoss(weight=weit, ignore_index=-1)
        self.att_a = SelfAttention1(dec_hidden_size)
        self.att_b = SelfAttention1(dec_hidden_size)
        self.lin_projection = nn.Sequential(
            nn.Linear(6 * dec_hidden_size, dec_hidden_size),
            nn.Dropout(drop_rate),
            nn.ReLU(True),
        )

    def generate_rep_via_att(self, cur_pred_mention_reps_, cur_pred_mention_embs_):

        men_rep_ = self.att_a(cur_pred_mention_reps_)
        men_reps = torch.cat((cur_pred_mention_reps_[0], men_rep_.squeeze(1), cur_pred_mention_reps_[-1]), -1)
        men_emb_ = self.att_a(cur_pred_mention_embs_)
        men_embs = torch.cat((cur_pred_mention_embs_[0], men_emb_.squeeze(1), cur_pred_mention_embs_[-1]), -1)

        men_repre = torch.cat((men_reps, men_embs), -1)
        men_repre = self.lin_projection(men_repre)

        return men_repre

    def forward(self, src_words_seq, src_words_seq_len, src_mask, src_char_seq, pos_seq, trg_seq, gd_y,
                is_training=False):

        batch_len = src_words_seq.size()[0]  # == 1
        src_time_steps = src_words_seq_len

        EOM_FLAG = 0
        NEXT_FLAG = src_time_steps - 1

        total_dec_time_steps = 0
        total_loss = 0.0

        ptr_trace = []
        ptr_input_trace = []

        pred_mentions = []
        cur_pred_mention_ = []

        cur_pred_mention_reps_ = []
        cur_pred_mention_embs_ = []

        enc_hs, src_word_embeds = self.encoder(src_words_seq, src_char_seq, pos_seq, is_training)

        h0 = autograd.Variable(torch.FloatTensor(torch.zeros(batch_len, dec_hidden_size)))
        c0 = autograd.Variable(torch.FloatTensor(torch.zeros(batch_len, dec_hidden_size)))
        if torch.cuda.is_available():
            h0 = h0.cuda()
            c0 = c0.cuda()

        dec_hid = (h0, c0)
        cur_zero_men_rep = autograd.Variable(torch.FloatTensor(torch.zeros(batch_len, dec_hidden_size)))
        cur_men_rep = autograd.Variable(torch.FloatTensor(torch.zeros(batch_len, dec_hidden_size)))
        psedu_emb = autograd.Variable(torch.FloatTensor(torch.zeros(batch_len, enc_inp_size)))
        psedu_prev_enc_hid = autograd.Variable(torch.FloatTensor(torch.zeros(batch_len, enc_hidden_size)))
        if torch.cuda.is_available():
            cur_zero_men_rep = cur_zero_men_rep.cuda()
            cur_men_rep = cur_men_rep.cuda()
            psedu_emb = psedu_emb.cuda()
            psedu_prev_enc_hid = psedu_prev_enc_hid.cuda()

        prev_dec_hid = torch.cat((cur_men_rep, psedu_emb, psedu_prev_enc_hid), -1)

        dec_outs = self.decoder(prev_dec_hid, dec_hid, enc_hs, cur_men_rep, src_mask, is_training)

        ptr_input_trace.append(-1)

        dec_hid = dec_outs[0]

        ptr_weights = dec_outs[1]
        ptr_index = ptr_weights.argmax(0).squeeze().data.cpu().numpy()
        last_ptr_index = ptr_index
        ptr_trace.append(ptr_index)

        if is_training:
            loss_ = self.myLoss(torch.log(ptr_weights.unsqueeze(0) + 1e-10), gd_y[:, 0])
            total_loss += loss_

        total_dec_time_steps += 1

        last_head_time_step = 0
        last_is_NEXT = True
        last_is_EOM = False

        cur_enc_time_step = trg_seq[1]
        first_NEXT = 1

        while cur_enc_time_step != src_time_steps - 1:
            ptr_input_trace.append(cur_enc_time_step)

            if is_training:
                gold_cur_enc_time_step = trg_seq[total_dec_time_steps]
                prev_dec_hid = torch.cat((cur_men_rep, src_word_embeds[:, gold_cur_enc_time_step, :],
                                          enc_hs[:, gold_cur_enc_time_step, :]), -1) + prev_dec_hid
            else:
                prev_dec_hid = torch.cat((cur_men_rep, src_word_embeds[:, cur_enc_time_step, :],
                                          enc_hs[:, cur_enc_time_step, :]), -1) + prev_dec_hid

            dec_outs = self.decoder(prev_dec_hid, dec_hid, enc_hs, cur_men_rep, src_mask, is_training)

            dec_hid = dec_outs[0]
            ptr_weights = dec_outs[1]

            if is_training:
                loss_ = self.myLoss(torch.log(ptr_weights.unsqueeze(0) + 1e-10), gd_y[:, total_dec_time_steps])
                total_loss += loss_

            ptr_index = ptr_weights.argmax(0).squeeze().data.cpu().numpy()

            if is_training:
                # teacher forcing on the route trace path
                ptr_index = gd_y[:, total_dec_time_steps].data.cpu().numpy().squeeze()

            ptr_trace.append(ptr_index)

            # path analyzing
            if ptr_index != NEXT_FLAG:

                if last_is_NEXT:
                    last_head_time_step = cur_enc_time_step

                if (NEXT_FLAG not in ptr_trace):
                    if first_NEXT == 2:
                        last_head_time_step = last_ptr_non_NEXT_index

                if last_is_NEXT or last_is_EOM:
                    cur_pred_mention_.append(cur_enc_time_step)
                else:
                    cur_pred_mention_.append(last_ptr_index)

                if last_is_EOM:
                    cur_pred_mention_.pop(len(cur_pred_mention_) - 1)

                if ptr_index != EOM_FLAG:
                    cur_pred_mention_reps_.append(dec_hid[0])
                    cur_pred_mention_embs_.append(enc_hs[:, cur_enc_time_step, :])

                    last_is_EOM = False

                elif ptr_index == EOM_FLAG:
                    last_is_EOM = True
                    pred_mentions.append(','.join([str(item) for item in cur_pred_mention_]))
                    cur_pred_mention_ = []
                    cur_pred_mention_reps_ = []
                    cur_pred_mention_embs_ = []

                cur_enc_time_step = ptr_index

                if len(cur_pred_mention_reps_) > 0:
                    men_repre = self.generate_rep_via_att(cur_pred_mention_reps_, cur_pred_mention_embs_)
                    cur_men_rep = men_repre
                else:
                    cur_men_rep = cur_zero_men_rep

                if first_NEXT == 1:
                    last_ptr_non_NEXT_index = ptr_index
                last_is_NEXT = False
                first_NEXT += 1

            elif ptr_index == NEXT_FLAG:
                if last_is_EOM:
                    cur_enc_time_step = (last_head_time_step + 1)
                else:
                    cur_enc_time_step += 1

                last_is_EOM = False
                last_is_NEXT = True

                cur_pred_mention_reps_ = []
                cur_pred_mention_embs_ = []
                cur_men_rep = cur_zero_men_rep

            total_dec_time_steps += 1
            last_ptr_index = ptr_index

            if is_training or True:
                cur_enc_time_step = trg_seq[total_dec_time_steps]

        if is_training:
            return total_loss
        else:
            return pred_mentions, ptr_input_trace, ptr_trace
