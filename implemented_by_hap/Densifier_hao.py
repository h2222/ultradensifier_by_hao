# coding = utf-8

# / -> //
from __future__ import division



import os
import itertools
import sys
import random

random.seed(3)
# 线程参数设置
os.environ['MKL_NUM_THREADS'] = '40'
os.environ['NUMEXPR_NUM_THREADS'] = '40'
os.environ['OMP_NUM_THREADS'] = '40'


#form helpers import normalizer, emblookup, emblookup_verbose, 
#                  line_process, word2vec

from sys import exit




def parse_words(add_bib=False):
    pos, neg = [], []
    with open('../po_ne_effect/myPos.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            if not add_bib:
                pos.append(line.strip())
            else:
                pos.append(line.strip())
    with open('../po_ne_effect/myNeg.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            if not add_bib:
                neg.append(line.strip())
            else:
                neg.append(line.strip())
    
    print(pos,'\n\n\n\n\n\n\n\n', neg)

    return pos, neg    








if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--LR", type=float, default=5., help="learning rate")
    parser.add_argument("--alpha", type=float, default=.5, help="hyper params balancing two sub-objective")
    parser.add_argument("--ECP", type=int, default=2., help="epoch")
    parser.add_argument("--OUT_DIM", type=int, default=1, help="output demension")
    parser.add_argument("--BATCH_SIZE", type=int, default=100, help="batch size")
    parser.add_argument("--EMB_SPACE", type=str, default='./path', help="input embedding space")
    parser.add_argument("--SAVE_EVERY", type=int, default=1000, help="save every N steps")
    parser.add_argument("--SAVE_TO", type=str, default='trained_densifier.pkl', help="output trained transformation matrix")


    pos_words, neg_words = parse_words(add_bib=False)
