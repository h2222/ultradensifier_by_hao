# coding = utf-8

# / -> //
from __future__ import division



import os
import itertools
import sys
import random
from scipy.stats import ortho_group
import pickle
import numpy as np
from random import randint

from my_helper import word2vec, emblookup

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
    
    # print(pos,'\n\n\n\n\n\n\n\n', neg)

    return pos, neg    


def batches(it, size):
    batch = []
    for item in it:
        batch.append(item)
        # generator, generating fixed size batch
        if len(batch) == size:
            yield batch
            batch = []
    
    # yield the last several items
    if len(batch) > 0:
        yield batch



class Densifier:

    def __init__(self, alpha, d, ds, lr, batch_size, seed=3):
        self.d = d  # input DIM (300)
        self.ds = ds  # output DIM (1)
        self.Q = np.matrix(ortho_group.rvs(d, random_state=seed))
        self.P = np.matrix(np.eye(ds, d))# [1, 300] 单位矩阵， 对角线为1其余为0  index matrix 
        self.D = np.transpose(self.P) * self.P # [1, 1] 1矩阵, D指word在原空间的表达[1, 300]经过正交转换后的在新的超密度空间中的表达[1, 1]
        self.zero_d = np.matrix(np.zeros((self.d, self.d))) # [300, 300] 0 矩阵
        self.lr = lr
        self.batch_size = batch_size
        self.alpha = alpha   # 超参数 alpha, 表示每一个任务的权重值 (sentiment, concreteness, frequency)   


    # loss 梯度更新函数
    def _gradient(self, loss, vec_diff):
        # loss值为0, 距离为0， 词与词之间没有距离
        # vec_diff [300, 1]
        if loss == 0.:
            print('Waring: check if there are replicated seed words!')
            return self.zero_d[0, :]

        # [300, 1] * [1, 300] * [300, 1]/loss
        # 梯度更新,  U S V^T 
        # S 为 对角矩阵
        # U 和 V 为正交变换策略,  UV^T 得出的 Q 为最接近
        return self.Q[0, :] * vec_diff * np.transpose(vec_diff) / loss

    def train(self, num_epoch, pos_vecs, neg_vecs, save_to, save_every):
        bs = self.batch_size
        save_step = 0

        # 差异性表达
        # product 函数制造元组,例:
        # itertools.prodcut([p1, p2, p3], [n1, n2])
        # 返回迭代器, 迭代 (p1, n1), (p1, n2), (p2, n1), (p2, n2), (p3, n1), (p3, n2) 
        diff_ps = [ i for i in itertools.product(pos_vecs, neg_vecs)]


        # 相似性表达
        # same process
        # combination组合迭代器, 输入一个数组和一个组合数, 返回所有组合
        # 返回迭代器迭代 (p1, p2) (p2, p3) (p3, p1)  + (n1, n2)
        same_ps = [ i for i in itertools.combinations(pos_vecs, 2)] + \
                    [i for i in itertools.combinations(neg_vecs, 2)]


        for e in range(num_epoch):

            # suffling
            random.shuffle(diff_ps)
            random.shuffle(same_ps)

            step_orth = 0
            step_print = 0
            step_same_loss, step_diff_loss = [], []

            # mini_diff format [(pv_a, nv_j), (pv_b, nv_c), ....] len = batch_sz(bs)
            # mini_same format [(pv_a, pv_b), (pv_c, pv_d),(nv_d, nv_f) ....] len = batch_sz(bs)
            for (mini_diff, mini_same) in zip(batches(diff_ps, bs), batches(same_ps, bs)):
                step_orth += 1
                step_print += 1
                save_step += 1
                diff_grad, same_grad = [], []

                EW, EV = [], []

                # ew format pv_a ew is the original space word representation of postive words
                # ev format nv_b ev is the original space word representation of negtive words

                # 差异处理 
                for ew, ev in mini_diff:
                    EW.append(np.asarray(ew))
                    EV.append(np.asarray(ev))

                # 正向词和负向词在原始空间中的向量表达见得欧式距离
                # 二维array 各维度相减相减
                # p-matrix EW [batch_sz, 300] - n-matrix EV [batch_sz, 300]
                VEC_DIFF = np.asarray(EW) - np.asarray(EV)


                # Q 正交矩阵 [300, 300]
                # 假设50 为 batch_sz
                # [50, 300] * [300, 1] 
                # DIFF_LOSS [50, 1]
                DIFF_LOSS = np.absolute(VEC_DIFF * self.Q[0, :].reshape(self.d, 1))

                for idx in range(len(EW)):
                    # 使用[0, 0] 因为DIFF_LOSS 返回的为 1x1 的矩阵取[0, 0]获取其对应值, 对应值为当前词之间的距离值
                    # 对于VEC 将取到的对于vector[1, 300] -> [300, 1]
                    # 对每一个词的diff_loss 进行梯度更新
                    diff_grad_step = self._gradient(DIFF_LOSS[idx][0, 0], 
                                                    VEC_DIFF[idx].reshape(self.d, 1))
                    # 所有词的梯度差异均值
                    diff_grad.append(diff_grad_step)


                # 相似处理
                EW, EV = [], []
                for ew, ev in mini_same:
                    EW.append(np.asarray(ew))
                    EV.append(np.asarray(ev))

                VEC_SAME = np.asarray(EW) - np.asarray(EV)
                SAME_LOSS = np.absolute(VEC_SAME * self.Q[0, :].reshape(self.d, 1))
                for idx in range(len(EW)):
                    # 计算相似梯度
                    same_grad_step = self._gradient(SAME_LOSS[idx][0, 0], VEC_SAME[idx].reshape(self.d, 1))
                    same_grad.append(same_grad_step)

                
                # [batch_size, 300]
                diff_grad = np.mean(diff_grad, axis = 0)
                same_grad = np.mean(same_grad, axis = 0)


                # lr * (-2α*Dg + 2(1-α)*Sg)
                self.Q[0, :] -= self.lr * (-1. * self.alpha * diff_grad * 2. + (1.-self.alpha) * 2.)


                step_same_loss.append(np.mean(SAME_LOSS))
                step_diff_loss.append(np.mean(DIFF_LOSS))


                if step_print % 10 == 0:
                    print("+" * 100)
                    try:
                        print("Diff-loss: {:4f}, Same-loss: {:4f}, LR: {:4f}"
                        .format(np.mean(step_diff_loss),
                                np.mean(step_same_loss),
                                self.lr))

                        print(np.sum(self.Q))
                    except:
                        print(np.mean(step_diff_loss))
                        print(np.mean(step_same_loss))
                        print(self.lr)

                    step_same_loss, step_diff_loss = [], []
                
                if step_orth % sys.maxsize == 0:
                    self.Q = Densifier.make_orth(self.Q)
                if save_step % save_every == 0:
                    self.save(save_to)
                    print ("Model saved! Step: {}".format(save_step))
            print ("="*25 + " one epoch finished! ({}) ".format(e) + "="*25)
            # 学习率减少至原来的0.99
            self.lr *= 0.99
        print("Training finished ...")
        self.save(save_to)
        


    def save(self, save_to):
        with open(save_to, 'wb') as f:
            pickle.dump(self.__dict__, f, protocol=0)
        
        print('Trained mode saved ...')
                    

    @staticmethod
    def make_orth(Q):
        U, _, V = np.linalg.svd(Q)
        return U * V



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--LR", type=float, default=5., help="learning rate")
    parser.add_argument("--alpha", type=float, default=.5, help="hyper params balancing two sub-objective")
    parser.add_argument("--ECP", type=int, default=2, help="epoch")
    parser.add_argument("--OUT_DIM", type=int, default=1, help="output demension")
    parser.add_argument("--BATCH_SIZE", type=int, default=100, help="batch size")
    parser.add_argument("--EMB_SPACE", type=str, default='../../wiki-news-300d-1M.vec', help="input embedding space")
    parser.add_argument("--SAVE_EVERY", type=int, default=1000, help="save every N steps")
    parser.add_argument("--SAVE_TO", type=str, default='./output/result.pickle', help="output trained transformation matrix")
    
    args = parser.parse_args()

    # # format [pos_word1, pos_word2, pos_word3, ...]
    # # format [neg_word1, neg_word2, neg_word3, ...]
    # pos_words, neg_words = parse_words(add_bib=False)
    
    # # format [(word1,  [vector1]), (word2, [vecor2], ....]
    # myword2vec = word2vec(args.EMB_SPACE)

    # print('finish loading embedding ....')

    # # suffling pos words, neg words
    # map(lambda  x: random.shuffle(x), [pos_words, neg_words])

    # # get pos/neg word vector from embedding table
    # # the tensorflow also provid tf.embedding_loopup(word_index, table)

    # # pos_vecs format [[posw_v1], [posw_v2], ....]
    # # neg_vecs format [[negw_v1], [megw_v2], ....]
    # pos_vecs, neg_vecs = map(lambda x: emblookup(x, myword2vec), [pos_words, neg_words])

    
    # using persudo data replace the true data
    pos_vecs = [[randint(1, 100) for i in range(10)] for i in range(500)]
    neg_vecs = [[randint(1, 100) for i in range(10)] for i in range(500)]



    #  if the pos / neg word not exists
    assert(len(pos_vecs)) > 0
    assert(len(neg_vecs)) > 0


    # 300 --> input dim
    # args.OUT_DIM --> p=ultradense space dim (output dim)
    # LR ---> learning rate
    # BATCH_SIZE --> 100
    mydensifier = Densifier(args.alpha, 10, args.OUT_DIM, args.LR, args.BATCH_SIZE)
    mydensifier.train(args.ECP, 
                      pos_vecs,
                      neg_vecs,
                      args.SAVE_TO,
                      args.SAVE_EVERY)



