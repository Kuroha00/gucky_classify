# -*- coding: utf-8 -*-
"""
学習
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

import tensorflow as tf

from utils import make_batchdata, make_dir, push_line
from makeCNNclass import CNN


def train():
    previous_epoch_num = int( input("previous epoch num: ") )
    train_epoch_num = int( input("train epoch num: ") )
    data_set = input("data set choice(all or keras or manually or only raw): ")
    
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
    
    # 小数化
    X_train_std = X_train / 255
    X_test_std = X_test / 255
    X_train_std = X_train_std.astype(np.float32)
    X_test_std = X_test_std.astype(np.float32)
    
    model = CNN()
    model.load(previous_epoch_num=previous_epoch_num, data_set=data_set)
    fig = model.train(train_data=(X_train_std, y_train), valid_data=(X_test_std, y_test), train_epoch_num=train_epoch_num, initialize=False)
    model.save(epoch=previous_epoch_num+train_epoch_num, data_set=data_set)
    
    plt.show()
    
if __name__ == "__main__":
    train()