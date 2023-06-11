from framework.utils.common import *
from framework.utils.util import *


def get_metrics(gold_mentions, pred_mentions):
    gt_ = 0
    pred_ = 0
    correct_ = 0

    for gts, pds in zip(gold_mentions, pred_mentions):
        gt_ += len(gts)
        pred_ += len(pds)
        for pd in pds:
            if pd in gts:
                correct_ += 1

    print(pred_, '\t|\t', gt_, '\t|\t', correct_)
    p = float(correct_) / (pred_ + 1e-18)
    r = float(correct_) / (gt_ + 1e-18)
    F1 = (2 * p * r) / (p + r + 1e-18)
    print('P:', p, 'R:', r, 'F1:', F1)

    return p, r, F1
