# -*- coding: utf-8 -*-
"""
CNNモデルを定義
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

import tensorflow as tf

from utils import make_batchdata, make_dir, push_line


class CNN():
    
    def __init__(self, batch_size=16, learning_rate=1e-5, dropout_rate=0.5):
        
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate  # 学習時にのみドロップアウト
        
        # グラフ定義
        g = tf.Graph()
        with g.as_default():
            self.build()
            self.init_op = tf.global_variables_initializer()
            self.saver = tf.train.Saver()
        
        self.sess = tf.Session(graph=g)
    
    
    def build(self):
        # CNN層の構成，ロスなど定義
        
        # placeholder: データは未定のままグラフを構築して，具体的な値は実行するときに与えられる．
        tf_x = tf.placeholder(tf.float32, [None, 128, 128, 3], name="tf_x")
        tf_y = tf.placeholder(tf.int32, [None], name="tf_y")
        tf_y_onehot = tf.one_hot(indices=tf_y, depth=2, dtype=tf.float32, name='input_y_onehot')
        is_train = tf.placeholder(tf.bool, shape=(), name='is_train')
        
        # 層の定義
        conv_h1 = tf.layers.conv2d(tf_x, kernel_size=(5,5), filters=32, activation=tf.nn.relu)
        h1_pool = tf.layers.max_pooling2d(conv_h1, pool_size=(2,2), strides=(2,2))
        
        conv_h2 = tf.layers.conv2d(h1_pool, kernel_size=(5,5), filters=64, activation=tf.nn.relu)
        h2_pool = tf.layers.max_pooling2d(conv_h2, pool_size=(2,2), strides=(2,2))
        
        # conv_h3 = tf.layers.conv2d(h2_pool, kernel_size=(5,5), filters=64, activation=tf.nn.relu)
        # h3_pool = tf.layers.max_pooling2d(conv_h3, pool_size=(2,2), strides=(2,2))
        
        # 全結合層
        input_shape = h2_pool.get_shape().as_list()  # 
        n_input_units = np.prod(input_shape[1:])  # 
        h3_pool_flat = tf.reshape(h2_pool, shape=[-1, n_input_units])
        h4 = tf.layers.dense(h3_pool_flat, 128, activation=tf.nn.relu)
        h4_drop = tf.layers.dropout(h4, rate=self.dropout_rate, training=is_train)  # is_train: Trainのときだけ0.5の確率でdropout
        
        # 全結合層　最終層
        h5 = tf.layers.dense(h4_drop, 2, activation=None)  # Noneの場合 線形になる
        
        # 損失
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=h5, labels=tf_y_onehot), name="cross_entropy_loss")
        
        # 最適化
        optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(loss, name="train_op")
        
        # evaluation
        probabilities = tf.nn.softmax(h5, name="probabilities")
        correct = tf.equal(tf.cast( tf.argmax(h5, axis=1), tf.int32), tf_y)  # tf.cast: 型変換
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")
    
    
    def train(self, train_data, valid_data, train_epoch_num, initialize=True):
        
        # initialize variables
        if initialize:
            self.sess.run(self.init_op)
        
        train_acc_list = []
        valid_acc_list = []
        X_train_std, y_train = train_data
        X_valid_std, y_valid = valid_data
        
        for epoch in range(1, train_epoch_num+1):
            batch_gen = make_batchdata(X=X_train_std, y=y_train, batch_size=self.batch_size, shuffle=True)
            avg_loss = 0
            for i, (batch_x, batch_y) in enumerate(batch_gen):
                if ( (i%100==0) and (not i==0) ):
                    print(i)
                
                feed_dict_train = {"tf_x:0": batch_x, "tf_y:0":batch_y, "is_train:0": True}  # train
                loss_tmp, _ = self.sess.run(["cross_entropy_loss:0", "train_op"], feed_dict=feed_dict_train)  # loss_tmpがlist
                avg_loss += loss_tmp
            # ここで学習は終了
            
            print( "Epoch {}: Training Avg Loss: {}".format(epoch, avg_loss) )
            
            print("Train")
            train_shape = X_train_std.shape[0]
            data_num = 50  # 学習後に予測結果を出すときのバッチデータ数
            tmp_train_acc_list = []
            batch_gen = make_batchdata(X=X_train_std, y=y_train, batch_size=data_num, shuffle=True)
            for i, (X_tmp, y_tmp) in enumerate(batch_gen):
                feed_dict_train = {"tf_x:0":X_tmp, "tf_y:0":y_tmp, "is_train:0":False}
                tmp_train_acc_list.append( self.sess.run("accuracy:0", feed_dict=feed_dict_train) * y_tmp.shape[0] )
            
            train_acc  = np.sum(tmp_train_acc_list) / train_shape
            train_acc_list.append( train_acc )
            print("Train Acc: {}".format(train_acc))
            
            
            print("Valid")
            valid_shape = X_valid_std.shape[0]
            tmp_valid_acc_list = []
            batch_gen = make_batchdata(X=X_valid_std, y=y_valid, batch_size=data_num, shuffle=True)
            for i, (X_tmp, y_tmp) in enumerate(batch_gen):
                feed_dict_test = {"tf_x:0":X_tmp, "tf_y:0":y_tmp, "is_train:0":False}
                tmp_valid_acc_list.append( self.sess.run("accuracy:0", feed_dict=feed_dict_test) * y_tmp.shape[0] )
            
            valid_acc  = np.sum(tmp_valid_acc_list) / valid_shape
            valid_acc_list.append( valid_acc )
            print("Valid Acc: {}".format( valid_acc ))
            print("\n")
            
            if epoch==1:
                try: push_line(message="finish epoch 1") # Line 送信
                except: pass
        
        try: push_line(message="finish {} epoch".format(train_epoch_num))  # Line送信
        except: pass
        
        # 正解率のグラフ生成
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot(train_acc_list, label="train", color="b")
        ax.plot(valid_acc_list, label="test", ls=":", color="r")
        ax.set_ylim(0, 1)
        plt.legend(loc="best")
        plt.tight_layout()
        # plt.show()
        
        return fig
    
    
    def save(self, save_epoch, data_set, path="./tflayers-model"):
        save_path = os.path.join(path, data_set)
        make_dir(save_path)
        self.saver.save(self.sess, os.path.join(save_path, "model.ckpt"), global_step=save_epoch)
    
    
    def load(self, previous_epoch_num, data_set, path="./tflayers-model"):
        load_path = os.path.join(path, data_set)
        self.saver.restore(self.sess, os.path.join(load_path, "model.ckpt-{}".format(previous_epoch_num)))
        
    
    def predict(self, x_test_data):
        # 予測ラベルとその予測確率を返す．
        feed_dict = {"tf_x:0": x_test_data, "is_train:0": False}
        return self.sess.run("probabilities:0", feed_dict=feed_dict)