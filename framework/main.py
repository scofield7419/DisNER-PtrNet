import sys
import os
import numpy as np
import warnings
from collections import OrderedDict
import datetime
import json
from tqdm import tqdm
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from framework.utils.data_util import *
from framework.utils.util import *
from framework.utils.eval import *
from framework.utils.common import *
from framework.core.seq2seq import Seq2SeqModel

torch.backends.cudnn.deterministic = True

warnings.filterwarnings('ignore')


def get_model(model_id):
    if model_id == 1:
        return Seq2SeqModel(word_vocab_len, char_vocab_len, word_embed_matrix)


def train_model(model_id, train_samples, dev_samples, best_model_file):
    train_size = len(train_samples)
    train_samples = train_samples[:int(train_size / 10)]
    train_size = len(train_samples)
    batch_count = int(math.ceil(train_size / batch_size))
    custom_print(batch_count)
    model = get_model(model_id)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    custom_print('Parameters size:', pytorch_total_params)

    custom_print(model)
    if torch.cuda.is_available():
        model.cuda()

    custom_print('weight factor:', wf)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=lr_rate_decay)
    custom_print(optimizer)

    best_dev_f = -1.0
    best_epoch_idx = -1
    for epoch_idx in range(0, num_epoch):
        model.train()
        model.zero_grad()
        custom_print('Epoch:', epoch_idx + 1)
        cur_seed = random_seed + epoch_idx + 1

        set_random_seeds(cur_seed)
        cur_shuffled_train_data = shuffle_data(train_samples)
        start_time = datetime.datetime.now()
        train_loss_val = 0.0

        optimizer = decay_learning_rate(optimizer, epoch_idx, learning_rate)

        for batch_idx in tqdm(range(0, batch_count)):
            batch_start = batch_idx * batch_size
            batch_end = min(len(cur_shuffled_train_data), batch_start + batch_size)
            cur_batch = cur_shuffled_train_data[batch_start:batch_end]

            outputs_loss = 0.0

            for ele in range(len(cur_batch)):
                src_words_seq_, src_words_mask_, src_chars_seq_, positional_seq_, \
                trg_seq_, gd_y_, gold_mention_, src_words_seq_len_ = get_variablized_data(cur_batch[ele], word_vocab,
                                                                                          char_vocab)

                outputs_loss += model(src_words_seq_, src_words_seq_len_, src_words_mask_, src_chars_seq_,
                                      positional_seq_,
                                      trg_seq_, gd_y_, True)

            outputs_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            if (batch_idx + 1) % update_freq == 0:
                optimizer.step()
                model.zero_grad()
            train_loss_val += outputs_loss.item()

        train_loss_val /= batch_count
        end_time = datetime.datetime.now()
        custom_print('\nTraining loss:', train_loss_val)
        custom_print('Training time:', end_time - start_time)

        custom_print('\nDev Results\n')
        set_random_seeds(random_seed)
        dev_metrics = predict(train_samples, model)

        dev_p, dev_r, dev_f = dev_metrics

        if dev_f >= best_dev_f:
            best_epoch_idx = epoch_idx + 1
            best_epoch_seed = cur_seed
            custom_print('\nmodel saved......')
            best_dev_f = dev_f
            torch.save(model.state_dict(), best_model_file)

        custom_print('\n\n')
        if epoch_idx + 1 - best_epoch_idx >= early_stop_cnt:
            break

    custom_print('*******')
    custom_print('Best Epoch:', best_epoch_idx)
    custom_print('Best Dev F1:', best_dev_f)


def predict(samples, model):
    model.eval()
    set_random_seeds(random_seed)
    start_time = datetime.datetime.now()

    gold_mentions = []
    pred_mentions = []

    gd_ptr_input_trace = []
    prd_ptr_input_trace = []
    gd_ptr_trace = []
    prd_ptr_trace = []
    for instance_idx in tqdm(range(0, len(samples))):
        src_words_seq_, src_words_mask_, src_chars_seq_, positional_seq_, \
        trg_seq_, gd_y_, gold_mention_, src_words_seq_len_ = get_variablized_data(samples[instance_idx], word_vocab,
                                                                                  char_vocab)
        gold_mentions.append(gold_mention_)

        with torch.no_grad():
            pred_mentions_, ptr_input_trace, ptr_trace = model(src_words_seq_, src_words_seq_len_, src_words_mask_,
                                                               src_chars_seq_, positional_seq_,
                                                               trg_seq_, gd_y_, False)

        pred_mentions.append(pred_mentions_)
        prd_ptr_input_trace.append(ptr_input_trace)
        prd_ptr_trace.append(ptr_trace)
        gd_ptr_input_trace.append(trg_seq_)
        gd_ptr_trace.append(gd_y_.squeeze().data.cpu().numpy())
        model.zero_grad()

    end_time = datetime.datetime.now()
    custom_print('\nPrediction time:', end_time - start_time)
    metrics = get_metrics(gold_mentions, pred_mentions)
    return metrics


def custom_print(*msg):
    for i in range(0, len(msg)):
        if i == len(msg) - 1:
            print(msg[i])
            logger.write(str(msg[i]) + '\n')
        else:
            print(msg[i], ' ', end='')
            logger.write(str(msg[i]))


if __name__ == "__main__":

    n_gpu = torch.cuda.device_count()
    set_random_seeds(random_seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    if job_mode == 'train':
        logger = open(os.path.join(trg_data_folder, 'training.log'), 'w')
        custom_print('loading data......')
        model_file_name = os.path.join(trg_data_folder, 'model.h5py')

        train_file = os.path.join(src_data_folder, 'train.txt')
        train_data = read_data(train_file, 1)

        dev_file = os.path.join(src_data_folder, 'dev.txt')
        dev_data = read_data(dev_file, 2)

        custom_print('Training data size:', len(train_data))
        custom_print('Development data size:', len(dev_data))
        custom_print("preparing vocabulary......")
        save_vocab = os.path.join(trg_data_folder, 'vocab.pkl')

        word_vocab, char_vocab, word_embed_matrix = build_vocab(train_data, save_vocab, embedding_file)
        word_vocab_len = len(word_vocab)
        char_vocab_len = len(char_vocab)
        # NEXT_FLAG = word_vocab['<NEXT>']
        # EOM_FLAG = word_vocab['<EOM>']

        custom_print("Training started......")
        train_model(model_name, train_data, dev_data, model_file_name)
        logger.close()

    if job_mode == 'test':
        logger = open(os.path.join(trg_data_folder, 'test.log'), 'w')
        custom_print(sys.argv)
        custom_print("loading word vectors......")
        vocab_file_name = os.path.join(trg_data_folder, 'vocab.pkl')
        word_vocab, char_vocab = load_vocab(vocab_file_name, char_vocab)
        # NEXT_FLAG = word_vocab['<NEXT>']
        # EOM_FLAG = word_vocab['<EOM>']

        word_embed_matrix = np.zeros((len(word_vocab), word_embed_dim), dtype=np.float32)
        custom_print('vocab size:', len(word_vocab))

        model_file = os.path.join(trg_data_folder, 'model.h5py')

        best_model = get_model(model_name)
        custom_print(best_model)
        if torch.cuda.is_available():
            best_model.cuda()
        if n_gpu > 1:
            best_model = torch.nn.DataParallel(best_model)
        best_model.load_state_dict(torch.load(model_file))

        custom_print('\nTest Results\n')
        test_file = os.path.join(src_data_folder, 'test.txt')
        test_data = read_data(test_file, 3)

        print('Test size:', len(test_data))
        set_random_seeds(random_seed)
        test_metrics = predict(test_data, best_model)
