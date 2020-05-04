#coding=utf-8


import os
import itertools
import tensorflow as tf
import numpy as np
import pandas as pd
import scipy.stats as st
from numpy.linalg import svd
from sklearn.model_selection import KFold

from Utils import (evall, average_results_df, Embedding, load_cnseed, scale_prediction_to_seed)


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

class Densifier:
    
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.d = self.embeddings.m.shape[1]   # word embedding 的维度
        self.P = np.zeros(shape=[self.d, 1])  # index matrix 
        self.P[0, 0] = 1 # 文章认为, transfer matrix Q 的第一个, 代表词的情感倾向
        self.seed_lexicon = None
        self.induced_lexicon = None
        self.Qs = None # 将情感倾向映射为 matrix 并存储在 Qs中
    


    def  train_lexicon(self, labels, save_path):
        self.fit(labels)
        self.induced_lexicon.to_csv(save_path, index=True, encoding='utf_8_sig')
    

    def fit(self,
            seed_lexicon, # training label, dataframe, [train_seed, sentiment]
            binarization_threshold=.5,
            alpha=.7):
        
        tf.reset_default_graph()
        self.seed_lexicon = seed_lexicon
        # index : vec word, columns = [sentiment]
        self.induced_lexicon = pd.DataFrame(columns=self.seed_lexicon.columns,
                                            index=self.embeddings.iw)

        # 二值化 pos/neg
        binarized_lexicon = self.binarize()


        print('='*100)
        #print(binarized_lexicon)


        # training per word
        self.train_Q(pos=binarized_lexicon['sentiment']['pos'],
                                    neg=binarized_lexicon['sentiment']['neg'],
                                    batch_size=100, #100
                                    optimizer='sgd',
                                    orthogonalize=True,
                                    alpha=alpha,
                                    training_steps=2000) #3000 x 100

        self.induced_lexicon['sentiment'] = self.embeddings.m.dot(self.Qs).dot(self.P)
        # self.induced_lexicon.to_csv('./step/step_save'+str(i_step)+'.csv', index=True, encoding='utf-8')
        # P*Q* m     m [vocab_size, dim]     Q [dim, dim]    P [dim, 1] 
        # induced_lexicon['sentiment']  s[batch_size, 1]

        #print('最终结果', self.induced_lexicon)
        print('最大值最小值词', 
              self.induced_lexicon.sort_values(by='sentiment', axis=0).head(100),
              '-------- \n', 
              self.induced_lexicon.sort_values(by='sentiment', axis=0, ascending=False).head(100))
        
        print('描述性统计', self.induced_lexicon.describe())        



   

    def vec(self, word):
        return self.embeddings.represent(word)


    def train_Q(self,
                pos, # 正项词index list 
                neg, # 负向词 index list
                alpha,
                batch_size,
                optimizer='sgd',
                orthogonalize=False,
                training_steps=4000):

        '''
        根据正向/负向词, 学习一个正交变换矩阵
        '''

        # 笛卡尔乘积 cartessian product of positive and negative seeds
        with tf.Graph().as_default():

            alpha = tf.constant(alpha, dtype=tf.float32)
            # 两两组合pos/neg word
            print('beginning to work on sperated pairs...')
            pairs_spearate = list(itertools.product(pos, neg))
            print('len data separate:', len(pairs_spearate))

            data_separate = pd.DataFrame(pairs_spearate)
            del pairs_spearate # 删除pair释放内存

            # 相似性组合
            print('beginning to work on aligned pairs...')
            pairs_align = list(itertools.combinations(pos, 2)) + \
                          list(itertools.combinations(neg, 2))

            print('Lenght of pairs_align:', len(pairs_align))
            data_align = pd.DataFrame(pairs_align)
            del pairs_align


            # 建立tensorflow graph
            # 引索矩阵 [dim_of_emb, 1] 其中第一个元素为1其他都为0
            P = tf.constant(self.P, dtype=tf.float32)
            # 正交矩阵 [dim_of_emb, dim_of_emb]
            Q = tf.Variable(tf.random_normal(shape=[self.d, self.d], stddev=1), name='Q')

            # 相似/不同 [batch_size, dim_of_emb] [6, 300]
            # e_w - e_v , 其中 w 和 v 来自不同的类
            e_diff = tf.placeholder(tf.float32, shape=[None, self.d], name='e_diff')
            # e_w - e_v , 其中 w 和 v 来自相同的类
            e_same = tf.placeholder(tf.float32, shape=[None, self.d], name='e_same')


            # loss function
            # QxP [300, 1]
            QxP = tf.matmul(Q, P)

            # 求和 [b, 300] * [300, 1] ==> [6, 1] 所有维度求和
            loss_separate = -tf.reduce_sum(tf.matmul(e_diff, QxP))
            loss_align = tf.reduce_sum(tf.matmul(e_same, QxP))

            # 损失函数
            loss = (alpha*loss_separate) + ((1 - alpha)*loss_align)


            ## define optimization

            ## Classical SGD 优化 随机梯度下降(according to paper)
            if optimizer == 'sgd':
                # global_step 的作用为能够保持全局步数的增加, 因为在学习率为指数衰减的学习率, 需要global step
                # 来保持休息率的更新, 通过minimize中的global_step 参数传入 expontila_decay 中 
                global_step = tf.Variable(0, trainable=False)
                starter_learning_rate=5.
                # 学习率指数下降
                learning_rate = tf.train.exponential_decay(
                    learning_rate=starter_learning_rate,
                    global_step=global_step,
                    decay_steps=10,
                    decay_rate=.99,
                    staircase=True
                )
                learning_step = (tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step))
            
            ## ADAM 优化器
            elif optimizer =='adam':
                learning_rate = tf.constant(1e-3)
                learning_step = (tf.train.AdamOptimizer(learning_rate).minimize(loss))
            else:
                raise NotImplementedError

            

            #开始计算
            with tf.Session() as sess:
                init = tf.global_variables_initializer()
                sess.run(init)

                # 保存器
                saver = tf.train.Saver(max_to_keep=1)

                gen_separate = Batch_Gen(data=data_separate, random=True, caller=self)
                gen_align = Batch_Gen(data=data_align, random=True, caller=self)

                # 复制正交矩阵Q
                last_Q = Q.eval()

                for i_step in range(training_steps):

                    if orthogonalize:
                        # re-orthogonalize matrix
                        u, s, v_T = svd(Q.eval())
                        new_q = u.dot(v_T.T)
                        # 不改变变量, 替换原有Q中的value
                        Q.assign(new_q).eval()
                    
                    # 一个batch的向量距离 [batch_sz, dim]
                    curr_separate = gen_separate.next(n=batch_size)
                    curr_align = gen_align.next(n=batch_size)
                    # 多计算运行 sess.run([a,b]) a 和 b 会同时计算, feed_dict会给placeholder
                    # e_diff 和 e_same 赋值
                    curr_loss, _ = sess.run([loss, learning_step],
                                            feed_dict={'e_diff:0':curr_separate,
                                                       'e_same:0':curr_align})
                    
                    if i_step%100 == 0:
                        curr_Q = Q.eval(session=sess)
                        Q_diff = np.sum(abs(last_Q - curr_Q))
                        print('eavluation: step:{0} , loss : {1}, lr:{2}, Q distance:{3}'.format(i_step, curr_loss, learning_rate.eval(), Q_diff))
                        last_Q = curr_Q
                
                print('Success')
                self.Qs = Q.eval() 
                    
                #saver.save(sess,'./saved/myModel_test')
                #print('Successfully saved the model')
               

    # 存储 + / - value
    def binarize(self):
        lexcion = self.seed_lexicon
        # save the +/- seed words
        binarized = {'sentiment':{'pos':[], 'neg':[]}}

        for i, x in lexcion.iterrows():
            if '+' == x['sentiment']:
                binarized['sentiment']['pos'] += [x.name.strip()]
            elif '-' == x['sentiment']:
                binarized['sentiment']['neg'] += [x.name.strip()]
    
        return binarized


class Batch_Gen:
    def __init__(self, data, caller, random=False):
        self.data = pd.DataFrame(data)
        self.index=0
        self.random = random
        self.len = self.data.shape[0]
        self.caller = caller # caller 指的是 densifier class 实例对象

    # 返回一个batch的向量距离差[batch_sz, 300]    
    def next(self, n):
        # 采样, 并减小data(防止重复采样)
        pairs = self.data.sample(n=n, axis=0, replace=True)        
        # [采样大小, dim]  初始化采样器
        batch = np.zeros([len(pairs), self.caller.d])

        for i in range(len(pairs)):
            #print('pairs', pairs)
            word1 = pairs.iloc[i][0]
            word2 = pairs.iloc[i][1]
            
            # ew - ev (欧几里得距离)
            batch[i] = self.caller.vec(word1) - self.caller.vec(word2)
        return batch




if __name__ == "__main__":
    
    # seed words
    #labels_total = load_cnseed(path='./Utils/source/cn_seed_v2.csv')
    labels_5 = load_cnseed(path='./Utils/source/cn_seed_v2_5.csv')
    labels_10 = load_cnseed(path='./Utils/source/cn_seed_v2_10.csv')
    labels_15 = load_cnseed(path='./Utils/source/cn_seed_v2_15.csv')
    labels = [labels_5, labels_10, labels_15]
    #labels = [labels_total]
    # embedding
    embs = []
    p, _, vecfs = next(os.walk('./Utils/source/vec'))
    #print(p)
    #print(vecfs[])

    # LOAD EMBEDDING
    for f in vecfs:
        p_f = p+'/'+f
        print(p_f)    
        emb = Embedding.from_fasttext_vec(path=p_f)
        print(f[:3])
        embs.append((emb, f[:3]))

    ## TRAIN
    for i, l in enumerate(labels):
        for emb, name in embs :
            densifier = Densifier(embeddings=emb)
            densifier.train_lexicon(l, save_path='./output/random_video/'+name+'_lexicon'+str(i*5)+'.csv')
            print('successful create '+str(name)+' lexicon by uising label:', i*5)
            del densifier




