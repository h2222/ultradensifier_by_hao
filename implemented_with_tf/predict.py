# coding = utf-8


import tensorflow as tf
from Utils import (evall, average_results_df, Embedding, load_cnseed, scale_prediction_to_seed)






def predict():
   
    sess = tf.Session()
    
    ckpt_path = './home/jiaxiang.hao/ultradensifier_model_save/saved/checkpoint'
    mate_path = './home/jiaxiang.hao/ultradensifier_model_save/saved/myModel.meta'
    model = tf.train.import_meta_graph(mate_path)
    model.restore(sess, tf.train.latest_checkpoint(chpt_path))

    graph = tf.get_default_graph()
    sess.run(tf.tables_initializer())




    














