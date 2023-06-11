from framework.utils.common import *
from framework.utils.util import *
import torch
import torch.autograd as autograd
from recordclass import recordclass
from collections import OrderedDict
import pickle

import os, sys
import numpy as np

Sample = recordclass("Sample", "Id SrcSeqs TrgSeqs TrgOutput Mention")


def read_data(data_file, datatype):
    '''
    :param data_file:
    :param datatype: 1: training data
    :return:
    '''
    # mode = {1:'train.txt', 2:'dev.txt', 3:'test.txt'}

    # with open(data_file, 'r') as reader:
    #     data_lines = reader.readlines()
    data_lines = make_data(data_file)

    samples = []
    uid = 1
    for i in range(0, len(data_lines)):
        src_line = data_lines[i].strip().split('|||')
        src_words = src_line[0].strip().split()

        tgt_pointers = src_line[1].strip().split()  # 第一个<S>: -1
        tgt_pointers = [int(item) for item in tgt_pointers]

        tgt_output_pointer = src_line[2].strip().split()  #
        tgt_output_pointer = [int(item) for item in tgt_output_pointer]

        mentions = src_line[3].strip().split('>')  #
        mentions = [item for item in mentions if item != '']
        # mentions = '|'.join(mentions)

        if datatype == 1 and (len(src_words) > max_src_len or len(tgt_pointers) > max_trg_len):
            continue

        sample = Sample(Id=uid, SrcSeqs=src_words, TrgSeqs=tgt_pointers,
                        TrgOutput=tgt_output_pointer, Mention=mentions)
        samples.append(sample)
        uid += 1

    return samples


def make_data(data_folder):
    max_len = 0
    max_tgt_len = []
    lens = []

    id_ = 0
    instances = []
    with open(data_folder, "r") as f:
        for sentence in f:
            # print(sentence)
            tokens = [t for t in sentence.strip().split()]
            tokens.insert(0, EOM)
            tokens.append(NEXT)

            lens.append(len(tokens))
            if len(tokens) > max_len:
                max_len = len(tokens)

            annotations = next(f).strip()
            # actions = self.parse.mention2actions(annotations, len(tokens))
            # oracle_mentions = [str(s) for s in self.parse.parse(actions, len(tokens))]
            gold_mentions = annotations.split("|") if len(annotations) > 0 else []
            gold_mentions_ = []
            # flag = False

            head_index = []
            for men in gold_mentions:
                indexe_ = men.strip().split()[0].strip().split(',')
                indexes = [int(item) + 1 for item in indexe_]
                assert len(indexes) % 2 == 0
                indexes = list(set(indexes))
                indexes.sort()
                head_index.append(indexes[0])
                # if len(indexes) > 4: flag = True
                gold_mentions_.append(indexes)

            gold_mentions_.sort()
            head_index = list(set(head_index))
            head_index.sort()
            # if flag:
            #     print(file_n, gold_mentions_)

            NEXT_pointer = len(tokens) - 1
            tgt_seq_pointers = []
            tgt_output_pointers = []
            tgt_seq_pointers.append(S_pointer)
            tgt_output_pointers.append(NEXT_pointer)
            for idx, item in enumerate(tokens):
                if idx == 0: continue  # skip the EOM token (as S)
                if idx in head_index:
                    for cur_men in gold_mentions_:
                        if cur_men[0] == idx:
                            for posi_ in range(0, len(cur_men) - 1):
                                tgt_seq_pointers.append(cur_men[posi_])
                                tgt_output_pointers.append(cur_men[posi_ + 1])
                            tgt_seq_pointers.append(cur_men[-1])
                            tgt_output_pointers.append(EOM_pointer)

                            tgt_seq_pointers.append(EOM_pointer)
                            tgt_output_pointers.append(idx)

                    tgt_output_pointers.pop(len(tgt_output_pointers) - 1)  # replace as NEXT
                    tgt_output_pointers.append(NEXT_pointer)

                else:
                    tgt_seq_pointers.append(idx)
                    tgt_output_pointers.append(NEXT_pointer)

            tokens_str = ' '.join(tokens)
            tgt_seq_pointers = ' '.join([str(item) for item in tgt_seq_pointers])
            tgt_output_pointers = ' '.join([str(item) for item in tgt_output_pointers])
            gold_mentions_str = '>'.join([','.join([str(item) for item in men]) for men in gold_mentions_])
            # assemble:
            instances.append(
                tokens_str + '|||' + tgt_seq_pointers + '|||' + tgt_output_pointers + '|||' + gold_mentions_str)

            assert len(next(f).strip()) == 0, f.readline()

            max_tgt_len.append(len(tgt_seq_pointers))

            id_ += 1

    return instances


def get_variablized_data(sample, word_vocab, char_vocab):
    src_words_seq_ = torch.from_numpy(np.array([get_words_index_seq(sample.SrcSeqs, word_vocab)]))
    src_words_mask_ = torch.from_numpy(np.array([get_padded_mask(len(sample.SrcSeqs))]))
    src_chars_seq_ = torch.from_numpy(np.array([get_char_seq(sample.SrcSeqs, char_vocab)]))
    positional_seq_ = torch.from_numpy(np.array([get_positional_index(len(sample.SrcSeqs))]))
    gd_y_ = torch.from_numpy(np.array([get_pointers(sample.TrgOutput)]))
    trg_seq_ = get_pointers(sample.TrgSeqs)

    if torch.cuda.is_available():
        src_words_seq_ = src_words_seq_.cuda()
        src_words_mask_ = src_words_mask_.cuda()
        src_chars_seq_ = src_chars_seq_.cuda()
        positional_seq_ = positional_seq_.cuda()
        gd_y_ = gd_y_.cuda()

    src_words_seq_ = autograd.Variable(src_words_seq_)
    src_words_mask_ = autograd.Variable(src_words_mask_)
    src_chars_seq_ = autograd.Variable(src_chars_seq_)
    positional_seq_ = autograd.Variable(positional_seq_)
    gd_y_ = autograd.Variable(gd_y_)

    gold_mentions = sample.Mention

    src_words_seq_len_ = len(sample.SrcSeqs)

    return src_words_seq_, src_words_mask_, src_chars_seq_, positional_seq_, \
           trg_seq_, gd_y_, gold_mentions, src_words_seq_len_


def shuffle_data(data):
    # custom_print(len(data))
    data.sort(key=lambda x: x.Id)
    # num_batch = int(len(data) / batch_size)
    # rand_idx = random.sample(len(data))#range(num_batch), num_batch)
    new_data = random.sample(data, len(data))
    return new_data


def get_words_index_seq(words, word_vocab):
    seq = list()
    for word in words:
        if word in word_vocab:
            seq.append(word_vocab[word])
        else:
            seq.append(word_vocab['<UNK>'])
    pad_len = max_src_len - len(words)
    for i in range(0, pad_len):
        seq.append(word_vocab['<PAD>'])
    return seq


def get_char_seq(words, char_vocab):
    char_seq = list()
    for i in range(0, conv_filter_size - 1):
        char_seq.append(char_vocab['<PAD>'])
    for word in words:
        for c in word[0:min(len(word), max_word_len)]:
            if c in char_vocab:
                char_seq.append(char_vocab[c])
            else:
                char_seq.append(char_vocab['<UNK>'])
        pad_len = max_word_len - len(word)
        for i in range(0, pad_len):
            char_seq.append(char_vocab['<PAD>'])
        for i in range(0, conv_filter_size - 1):
            char_seq.append(char_vocab['<PAD>'])

    pad_len = max_src_len - len(words)
    for i in range(0, pad_len):
        for i in range(0, max_word_len + conv_filter_size - 1):
            char_seq.append(char_vocab['<PAD>'])

    return char_seq


def get_padded_mask(cur_len):
    mask_seq = list()
    for i in range(0, cur_len):
        mask_seq.append(0)
    pad_len = max_src_len - cur_len
    for i in range(0, pad_len):
        mask_seq.append(1)
    return mask_seq


def get_positional_index(sent_len):
    index_seq = [min(i + 1, max_positional_idx - 1) for i in range(sent_len)]
    index_seq += [0 for _ in range(max_src_len - sent_len)]
    return index_seq


def get_pointers(seqs):
    pointer_list = []
    for item in seqs:
        pointer_list.append(item)
    return pointer_list


def load_word_embedding(embed_file, vocab):
    # custom_print('vocab length:', len(vocab))
    embed_vocab = OrderedDict()
    embed_matrix = list()

    embed_vocab['<PAD>'] = 0
    embed_matrix.append(np.zeros(word_embed_dim, dtype=np.float32))

    embed_vocab['<UNK>'] = 0
    embed_matrix.append(np.random.uniform(-0.25, 0.25, word_embed_dim))

    word_idx = 2
    with open(embed_file, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) < word_embed_dim + 1:
                continue
            word = parts[0]
            if word in vocab and vocab[word] >= word_min_freq:
                vec = [np.float32(val) for val in parts[1:]]
                embed_matrix.append(vec)
                embed_vocab[word] = word_idx
                word_idx += 1

    for word in vocab:
        if word not in embed_vocab and vocab[word] >= word_min_freq:
            embed_matrix.append(np.random.uniform(-0.25, 0.25, word_embed_dim))
            embed_vocab[word] = word_idx
            word_idx += 1

    # custom_print('embed dictionary length:', len(embed_vocab))
    return embed_vocab, np.array(embed_matrix, dtype=np.float32)


def build_vocab(data, save_vocab, embedding_file):
    vocab = OrderedDict()
    char_v = OrderedDict()
    char_v['<PAD>'] = 0
    char_v['<UNK>'] = 1
    char_idx = 2
    for d in data:
        for word in d.SrcSeqs:
            if word not in vocab:
                vocab[word] = 1
            else:
                vocab[word] += 1

            for c in word:
                if c not in char_v:
                    char_v[c] = char_idx
                    char_idx += 1

    word_v, embed_matrix = load_word_embedding(embedding_file, vocab)
    output = open(save_vocab, 'wb')
    pickle.dump([word_v, char_v], output)
    output.close()
    return word_v, char_v, embed_matrix


def load_vocab(vocab_file, char_vocab):
    with open(vocab_file, 'rb') as f:
        embed_vocab, char_vocab = pickle.load(f)
    return embed_vocab, char_vocab

# if __name__ == '__main__':
#     main()
