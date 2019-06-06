# -*- coding: utf-8 -*-
"""
学習
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

import cv2
import tensorflow as tf

from utils import make_dir, face_recognition_and_reshape
from makeCNNclass import CNN


def test():
    previous_epoch_num = int( input("previous epoch num: ") )
    data_set = input("data set choice(all or keras or manually or only raw): ")
    
    # 画像さんサンプル取得
    test_path = "./data/test_sample/"
    input_data = os.listdir(test_path)
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt2.xml')
    
    
    X_all = []
    for file in input_data:
        print("file: ", file)
        X = face_recognition_and_reshape(filename=test_path+file, face_cascade=face_cascade)
        if X is None:
            print("miss face recognition")
            continue
        X_std = X / 255
        X_all.append(X_std)
    
    X_all = np.array(X_all)
    X_all = X_all.astype(np.float32)
    
    model = CNN()
    model.load(previous_epoch_num=previous_epoch_num, data_set=data_set)
    prediction = model.predict(x_test_data=X_all)
    print("prediction: ", prediction)
    
    label = np.argmax(prediction, axis=1)  # 1がグッキー
    proba_list = []
    for i, arr in enumerate(prediction):
        proba_list.append(arr[label[i]])
        
    print("prediction result {} ( proba: {} )".format(label, proba_list) )


if __name__ == "__main__":
    test()