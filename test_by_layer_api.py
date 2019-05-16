# -*- coding: utf-8 -*-
"""
学習済みモデルに(128, 128)の画像データを
入力して確率を返すスクリプト
"""
import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

import tensorflow as tf
import cv2

from utils import make_dir, face_recognition_and_reshape


def test():
    epoch_num = int( input("previous epoch num: ") )  # 前回のモデルのエポック数    
    data_set = input("data set choice(all or keras or manually or only raw): ")
    
    
    # 画像さんサンプル取得
    test_path = "./data/test_sample/"
    input_data = os.listdir(test_path)
    
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt2.xml')
    
    pre_list = []
    for file in input_data:
        X = face_recognition_and_reshape(filename=test_path+file, face_cascade=face_cascade)
        pre_label, proba = prediction(X=X, epoch_num=epoch_num, data_set=data_set)
        print("{} prediction result {} (proba: {})".format(file, pre_label, proba) )
        # pre_list.append(pre_)
    
    
def prediction(X, epoch_num, data_set):
    """
    画像のパスを引数としてそのprediction結果を返す
    
    Parameters
    ----------
    X: array
    epoch_num: int
    data_set: str どのデータを使うか
    """
    X_std = X / 255
    X_std = X_std.astype(np.float32)
    

    with tf.Session() as sess:
        saver = tf.train.Saver()
        
        if not epoch_num == "":
            load_path = os.path.join("./tflayers-model", data_set)
            saver.restore(sess, os.path.join(load_path, 'model.ckpt-{}'.format(epoch_num) ))
        else:
            raise ValueError("epoch num")
        
        prediction = sess.run("probabilities:0", feed_dict={"tf_x:0": X_std})
        label = np.argmax(prediction)  # 1がグッキー

        
    return label, prediction[label]
        
    
if __name__ == "__main__":
    test()