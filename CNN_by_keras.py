# -*- coding: utf-8 -*-
"""
CNNモデルをkerasで定義
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

import tensorflow as tf

import keras
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D
from keras.optimizers import Adam
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.utils import np_utils  # One-Hot表現をするために使用

from utils import make_batchdata, make_dir, push_line


def build_model(dropout=0.5, output_class=2, learning_rate=1e-4):
    # kerasによるモデル構築
    model = Sequential()
    
    model.add(Conv2D(32, 3, input_shape=(128, 128, 3)))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    
    model.add(Conv2D(64, 3))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    
    model.add( Flatten() )  # 入力を平滑化　(None, 64, 32, 32)  ->  (None, 65536)
    model.add( Dense(128) )  # 全結合ネットワーク
    model.add( Activation('relu') )
    model.add( Dropout(dropout) )
    
    model.add( Dense(output_class, activation='softmax') )  # ラスト
    adam = Adam(lr=learning_rate)
    
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=["accuracy"])
    
    
    return model


def main():
    batch_size = 16
    output_class = 2
    
    # Xtrain, y_trainの取得
    previous_epoch = input("previous epoch num: ")
    train_epoch = input("train epoch num: ")
    data_set = input("data set choice(all or keras or manually or only raw): ")
    path = "./keras-model"
    model_path = os.path.join(path, data_set, "")
    
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
    
    X_train, y_train, _ = np.load(train_path)
    X_test, y_test, _ = np.load(test_path)
    y_train = np_utils.to_categorical(y_train, output_class)
    y_test = np_utils.to_categorical(y_test, output_class)
    
    # 小数化
    X_train_std = X_train / 255
    X_test_std = X_test / 255
    X_train_std = X_train_std.astype(np.float32)
    X_test_std = X_test_std.astype(np.float32)
    
    try:
        # モデルの読み込み
        if not previous_epoch=="":
            print("load model")
            model = keras.models.load_model(model_path+'model_{}.h5'.format(previous_epoch), compile=True)
        else:
            model = build_model(output_class=output_class)
        
        print("Train")
        history = model.fit(X_train_std, y_train, batch_size=batch_size, epochs=int(train_epoch), verbose=1, validation_split=0.1)
        
        # テスト
        print("Test")
        score = model.evaluate(X_test_std, y_test, verbose=1, batch_size=batch_size)
        print( list(zip(model.metrics_names, score)) )
        
        # モデル保存
        if not previous_epoch=="":
            save_epoch = int(previous_epoch) + int(train_epoch)
        else:
            save_epoch = int(train_epoch)
        
        save_model = model_path + "model_{}.h5".format(save_epoch)        
        model.save(save_model)
        
        # LINE出力
        push_line(message="finish {} epoch".format(train_epoch))
        
    except:
        print("error")
        push_line(message="error")
    
    
if __name__ == "__main__":
    main()