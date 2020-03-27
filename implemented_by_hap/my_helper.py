# coding = utf-8

from __future__ import division


from six.moves import xrange
from sys import exit
from multiprocessing import pool

import numpy as np
import scipy
import math







def line_process(l):
    pass



def word2vec(emb_path):
    word2vect = {} # 返回字典
    pool = Pool(4)
    # 多线程处理
    # [token,idx=1], [vector],[vector],[vector]....
    with open(emb_path, 'r', encoding='utf-8') as f:
        pairs = pool.map(line_process, f.readlines()[1:])
    pool.close()
    pool.join()
    _pairs = []
    for p in pairs:
        if p[0] is not None:
            _pairs.append(p)
    return  dict(_pairs)



def read(emb_path):
    with open(emb_path, 'rb+') as f:
        
        for i in f:
            try:
                print(i.decode('utf-8'))
            except:
                continue


if __name__ == "__main__":
    read('../wiki-news-300d-1M.vec')

