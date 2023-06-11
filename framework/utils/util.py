import framework.main as main
import random
import numpy as np
import torch
from framework.utils.common import *


def set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # if main.n_gpu > 1:
    #     torch.cuda.manual_seed_all(seed)


def save_output_dev(gold_mentions, pred_mentions, outfile):
    writer = open(outfile, 'w')
    for i in range(0, len(gold_mentions)):
        gold_mention_ = gold_mentions[i]
        pred_mention_ = pred_mentions[i]

        cur_str = 'golds: ' + '\t'.join('[' + item + ']' for item in gold_mention_)
        cur_str += "\n"
        cur_str += 'preds: ' + '\t'.join('[' + item + ']' for item in pred_mention_)
        cur_str += "\n\n"

        writer.write(cur_str)
    writer.close()


def save_trace_dev(trg_seq, prd_ptr_input_trace, gd_ptr_trace, prd_ptr_trace, outfile):
    writer = open(outfile, 'w')
    for i in range(0, len(trg_seq)):
        trg_seq_ = trg_seq[i]
        prd_trg_seq_ = prd_ptr_input_trace[i]

        gd_output_ = gd_ptr_trace[i]
        prd_output_ = prd_ptr_trace[i]

        cur_str = 'input golds: ' + '\t'.join('[' + str(item) + ']' for item in trg_seq_)
        cur_str += "\n"
        cur_str += 'input preds: ' + '\t'.join('[' + str(item) + ']' for item in prd_trg_seq_)
        cur_str += "\n------\n"
        cur_str += 'output golds: ' + '\t'.join('[' + str(item) + ']' for item in gd_output_)
        cur_str += "\n"
        cur_str += 'output preds: ' + '\t'.join('[' + str(item) + ']' for item in prd_output_)
        cur_str += "\n\n"

        writer.write(cur_str)
    writer.close()


def decay_learning_rate(optimizer, iter, init_lr):
    lr = init_lr / (1 + lr_rate_decay * iter)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer
