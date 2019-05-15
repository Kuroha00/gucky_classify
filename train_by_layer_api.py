# -*- coding: utf-8 -*-
"""
tensorflowのlayerAPIを用いて学習を回す
"""
import numpy as np
# import pandas as pd
import os
import sys

import tensorflow as tf

# from sklearn.preprocessing import StandardScaler

from utils import make_batchdata, make_dir


def main():
    do_filepath = __file__
    do_filename = os.path.basename(do_filepath)  # train_by_layer_api.
    
    dropout_rate = 0.5
    learning_rate = 1e-5
    epoch_num = 30
    batch_size = 16
    
    data_set = input("data set choice(all or keras or manually or only raw): ")
    
    # tensorflow定義
    g = tf.Graph()
    
    # データは未定のままグラフを構築して，具体的な値は実行するときに与えられる．
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
    input_shape = h2_pool.get_shape().as_list()
    n_input_units = np.prod(input_shape[1:])
    h3_pool_flat = tf.reshape(h2_pool, shape=[-1, n_input_units])
    h4 = tf.layers.dense(h3_pool_flat, 128, activation=tf.nn.relu)
    h4_drop = tf.layers.dropout(h4, rate=dropout_rate, training=is_train)  # is_train: Trainのときだけ0.5の確率でdropout
    
    # 全結合層　最終層
    h5 = tf.layers.dense(h4_drop, 2, activation=None) # TODO 
    
    # 損失
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=h5, labels=tf_y_onehot), name="cross_entropy_loss")
    
    # 最適化
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, name="train_op")
    
    # evaluation
    probabilities = tf.nn.softmax(h5)
    correct = tf.equal(tf.cast( tf.argmax(h5, axis=1), tf.int32), tf_y)  # tf.cast: 型変換
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")
    
    
    # 初期化
    init = tf.global_variables_initializer()
    # saver = tf.train.Saver()
    
    with tf.Session() as sess:
        sess.run(init)
        original_path = "data/input"
        if data_set == "all":
            train_path = os.path.join(original_path, "train_all_data.npy")
            test_path = os.path.join(original_path, "test_all_data.npy")
        elif data_set == "keras":
            train_path = os.path.join(original_path, "train_generated_by_keras.npy")
            test_path = os.path.join(original_path, "test_generated_by_keras.npy")
        elif data_set == "manually":
            train_path = os.path.join(original_path, "train_generated_manually.npy")
            test_path = os.path.join(original_path, "test_generated_manually.npy")
        elif data_set == "only raw":
            train_path = os.path.join(original_path, "train_only_raw.npy")
            test_path = os.path.join(original_path, "test_only_raw.npy")
        else:
            raise ValueError
        
        # データロード
        X_train, y_train, _ = np.load(train_path)
        X_test, y_test, _ = np.load(test_path)
        
        # 小数化
        X_train_std = X_train / 255
        X_test_std = X_test / 255
        
        train_accuracy_list = []
        test_accuracy_list = []
        for epoch in range(1, epoch_num+1):
            batch_gen = make_batchdata(X=X_train_std, y=y_train, batch_size=batch_size, shuffle=True)
            
            avg_loss = 0
            for i, (batch_x, batch_y) in enumerate(batch_gen):
                if ( (i%100==0) and (not i==0) ):
                    print(i)
                
                feed_dict_train = {"tf_x:0": batch_x, "tf_y:0":batch_y, "is_train:0": True}
                
                loss_tmp, _ = sess.run(["cross_entropy_loss:0", "train_op"], feed_dict=feed_dict_train)  # loss_tmpがlist
                avg_loss += loss_tmp
            
            
            print( "Epoch {}: Training Avg Loss: {}".format(epoch, avg_loss) )
            
            # print("Training Test")
            # feed_dict_valid = {"tf_x:0":X_train_std, "tf_y:0":y_train, "is_train:0": False}
            # train_acc = sess.run("accuracy:0", feed_dict=feed_dict_valid)
            
            # train_accuracy_list.append(train_acc)
            # print("Train Acc: {}".format(train_acc))
            
            print("Test")
            feed_dict_test = {"tf_x:0":X_test_std, "tf_y:0":y_test, "is_train:0": False}
            test_acc = sess.run("accuracy:0", feed_dict=feed_dict_test)
            
            test_accuracy_list.append(test_acc)
            print("Test Acc: {}".format(test_acc))
            print("\n")
        
        print("Test Accuracy List: ", test_accuracy_list)
        saver = tf.train.Saver()
        save_path = os.path.join("./tflayers-model", original_path, data_set)
        make_dir(save_path)
        saver.save(sess, os.path.join(save_path, "model.ckpt"), global_step=epoch_num)
    

if __name__ == "__main__":
    main()