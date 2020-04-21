#coding=utf-8
import itertools
import tensorflow as tf
import numpy as np
import pandas as pd
import scipy.stats as st
from numpy.linalg import svd
from sklearn.model_selection import KFold

from Utils import (evall, average_results_df, Embedding, load_anew99, scale_prediction_to_seed)


class Densifier:
    
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.d = self.embeddings.m.shape[1]   # word embedding 的维度
        self.P = np.zeros(shape=[self.d, 1])  # index matrix 
        self.P[0, 0] = 1 # 文章认为, transfer matrix Q 的第一个, 代表词的情感倾向
        self.seed_lexicon = None

        self.Qs = {} # 将情感倾向 VAD 映射为 matrix 并存储在 Qs中
    
    def fit(self,
            seed_lexicon, # training label, dataframe, [vocab_size(927), VAD(3)]
            binarization_threshold=.5,
            alpha=.7):
        
        tf.reset_default_graph()
        # print(seed_lexicon)

        self.seed_lexicon = seed_lexicon

        # index = embedding vocab.size=6, columns = [V, A, D]
        # [6, 3] 初始化
        self.induced_lexicon = pd.DataFrame(columns=self.seed_lexicon.columns,
                                            index=self.embeddings.iw)

        # 二值化 pos/neg
        binarized_lexicon = self.binarize(sd_threshold=binarization_threshold)


        print('='*100)
        print(binarized_lexicon)


        # training per word
        # var V A D 
        for var in list(self.induced_lexicon):
            print(var)
            self.Qs[var] = self.train_Q(pos=binarized_lexicon[var]['pos'],
                                        neg=binarized_lexicon[var]['neg'],
                                        batch_size=100, #100
                                        optimizer='sgd',
                                        orthogonalize=True,
                                        alpha=alpha,
                                        training_steps=1000) #3000

            # P*Q* m     m [batch_sz, dim]     Q [dim, dim]    P [dim, 1] 
            # induced_lexicon[var]  shape 为[batch_size, 1]

            # 分别计算计算和训练 V A D 对每个词
            self.induced_lexicon[var] = self.embeddings.m.dot(self.Qs[var]).dot(self.P)
    
    def predict(self, words):

        print('--'*30)
        print(self.induced_lexicon)

        # preds dataframe 行size = vocab size , 列 = V A D
        preds = self.induced_lexicon.loc[words]        
        # 如果字没有在 原始word embedding 中, 使用均值代替
        # V A D 分别为均值的 series
        means = self.induced_lexicon.mean(axis=0)

        for word in words:
            if not word in self.induced_lexicon.index:
                preds.loc[word] = means

        ### 删除pandas dataframe中index重复的行
        #   参数keep表示保留第一次出现的重复值
        preds = preds[~preds.index.duplicated(keep='frist')]

        ### rescalling pred size
        preds = scale_prediction_to_seed(preds=preds,
                                         seed_lexicon=self.seed_lexicon)

        return preds


    def eval_densifier(self, gold_lex):
        # inducd_lexicon 初始化 dataframe, 行size为vocab_size, 列为 VAD
        if self.induced_lexicon is None:
            raise ValueError('Embedding need to be transformed first! Run "fit"!')
        else:
            # 使用验证集取做预测

            print('8'*60)
            print(gold_lex)
            return(evall(gold_lex, self.predict(gold_lex.index)))

    

    def crossvalidate(self, labels, k_folds=10):
        '''
        labels : dataframe, axis0 = vacabularies , axis1 = V, A, D
        '''

        # 结果指标 V A D
        result_df = pd.DataFrame(columns=labels.columns)


        # KFold 函数, n_split 将label划分k_folds个互斥子集, 

        # 例如 labels的shape 为 [1030, 3] 1030 是 vocab_size, 3为VAD三个维度
        # 1030 分为n_splits=10份, 每份103,  train为9份为(927, 3) test为1份为(103, 3)
        # 交叉验证n_splits=10次
        kf = KFold(n_splits=k_folds, shuffle=True).split(labels)

        # print(next(kf))
        for i, split in enumerate(kf):
            # split = next(kf)
            train = labels.iloc[split[0]]
            test = labels.iloc[split[1]]
            print('训练次数', i) # 打印训练次数
            self.fit(train) # train 为[927, 3] 的df
            
            #返回结果, 交叉验证
            result_df.loc[k] = self.eval_densifier(test)
            result_df = average_results_df(result_df)
        result_df.to_excel('./result_df.xlsx', sheet_name=0, index='False', encoding='utf-8')
            




    def vec(self, word):
        return self.embeddings.represent(word)

    def train_Q(self,
                pos, # 正项词index list 
                neg, # 负向词 index list
                alpha,
                batch_size,
                optimizer='sgd',
                orthogonalize=True,
                training_steps=4000):

        '''
        根据正向/负向词, 学习一个正交变换矩阵
        '''

        # 笛卡尔乘积 cartessian product of positive and negative seeds
        with tf.Graph().as_default():

            alpha = tf.constant(alpha, dtype=tf.float32)
            # 两两组合pos/neg word
            pairs_spearate = list(itertools.product(pos, neg))
            print('len data separate:', len(pairs_spearate))

            data_separate = pd.DataFrame(pairs_spearate)
            del pairs_spearate # 删除pair释放内存

            # 相似性组合
            print('beginning to work on aligned pairs...')
            pairs_align = list(itertools.combinations(pos, 2)) + \
                          list(itertools.combinations(neg, 2))
            # pairs_align = combinations(pos) + combinations(neg)

            print('Lenght of pairs_align:', len(pairs_align))
            data_align = pd.DataFrame(pairs_align)
            del pairs_align


            # 建立tensorflow graph
            # 引索矩阵 [dim_of_emb, 1] [300, 1] 其中第一个元素为1其他都为0
            P = tf.constant(self.P, dtype=tf.float32)
            # 正交矩阵 [dim_of_emb, dim_of_emb] [300, 300]
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
                    decay_steps=1,
                    decay_rate=.99,
                    staircase=True
                )
                learning_step = (tf.train.GradientDescentOptimizer(learning_rate).
                minimize(loss, global_step=global_step))
            
            ## ADAM 优化器
            elif optimizer =='adam':
                learning_rate = tf.constant(1e-3)
                learning_step = (tf.train.AdamOptimizer(learning_rate).
                minimize(loss))
            else:
                raise NotImplementedError

            

            #开始计算
            with tf.Session() as sess:
                init = tf.global_variables_initializer()
                sess.run(init)

                gen_separate = Batch_Gen(data=data_separate, random=True, caller=self)
                gen_align = Batch_Gen(data=data_align, random=True, caller=self)

                # 复制正交矩阵Q
                last_Q = Q.eval()

                for i_step in range(training_steps):

                    if orthogonalize:
                        # re-orthogonalize matrix
                        # SVD 详解
                        # https://www.cnblogs.com/pinard/p/6251584.html
                        # Q [dim, dim]
                        # u [dim, k]  vT[dim, k]
                        # newQ u * vTT  [dim, dim]
                        u, s, v_T = svd(Q.eval())
                        new_q = u.dot(v_T.T)
                        # 不改变变量, 替换原有Q中的value
                        Q.assign(new_q).eval()
                    
                    # 一个batch的向量距离差 [batch_sz, dim(300)]
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
                        print('eavluation:',i_step, curr_loss, learning_rate.eval(), Q_diff)
                        last_Q = curr_Q
                
                print('Success')
                return Q.eval()

    # 二值化话
    def binarize(self, sd_threshold):
        # [927, 3] [train_label_vocab_size, VAD]
        lexcion = self.seed_lexicon

        # label的均值标准差(series 类型)
        means = lexcion.mean(axis=0)
        sds = lexcion.std(axis=0)

        print('lexcion shape:', lexcion.shape)
        print(means, sds)

        # var is V A D
        binarized = {var:{'pos':[], 'neg':[]} for var in list(lexcion)}
        print(binarized)

        # i_var is word
        for i_word in range(len(lexcion)):
            for i_var in range(len(list(lexcion))):
                var = list(lexcion)[i_var]
                mean = means.iloc[i_var]
                sd = sds.iloc[i_var]
                print('current var:', i_var)
                
                # 如果 VAE的值大于 对应VAE值的 均值 + 阈值 * 标准差, 为正项词语(pos)
                # 反之为负向词语(neg)
                if lexcion.iloc[i_word, i_var] > (mean + sd_threshold*sd):
                    binarized[var]['pos'] += [i_word]
                elif lexcion.iloc[i_word, i_var] < (mean - sd_threshold*sd):
                    binarized[var]['neg'] += [i_word]
        
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
        # 行采样, 并减小data
        pairs = self.data.sample(n=n, axis=0, replace=True)        
        # [采样大小, dim(3000)] # 初始化采样器
        batch = np.zeros([len(pairs), self.caller.d])

        # print('打印pairs', pairs)
        for i in range(len(pairs)):
            word1 = self.caller.seed_lexicon.index[pairs.iloc[i, 0]]
            word2 = self.caller.seed_lexicon.index[pairs.iloc[i, 1]]
            # ew - ev
            batch[i] = self.caller.vec(word1) - self.caller.vec(word2)
        return batch




if __name__ == "__main__":
    emb = Embedding.from_fasttext_vec(path='./Utils/densifier_test.vec')
    labels = load_anew99(path='./Utils/anew99.csv')
    print(labels.shape)    
    densifier = Densifier(embeddings=emb)
    densifier.crossvalidate(labels=labels)











