import os

S = '<S>'
S_pointer = -1
EOM = '<EOM>'
EOM_pointer = 0
NEXT = '<NEXT>'

random_seed = 42

src_data_folder = r'./data/examples'
trg_data_folder = r'../output/exp1'
if not os.path.exists(trg_data_folder):
    os.mkdir(trg_data_folder)

model_name = 1
job_mode = 'train'
batch_size = 1
num_epoch = 30
lr_rate_decay = 0.05
learning_rate = 0.00025

max_src_len = 100
max_trg_len = 300
w2v_file_path = r'../emb'
embedding_file = os.path.join(w2v_file_path, 'w2v.txt')
update_freq = 1
wf = 1.0
att_type = 0

use_hadamard = False
gcn_num_layers = 3
enc_type = 'LSTM'

word_embed_dim = 300
word_min_freq = 1

char_embed_dim = 50
char_feature_size = 50
conv_filter_size = 3
max_word_len = 10
positional_embed_dim = 20
max_positional_idx = 100

enc_inp_size = word_embed_dim + char_feature_size + positional_embed_dim
enc_hidden_size = word_embed_dim
dec_inp_size = enc_hidden_size
dec_hidden_size = dec_inp_size
l1_type_embed_dim = 50

drop_rate = 0.3
layers = 2
early_stop_cnt = 5
sample_cnt = 0
