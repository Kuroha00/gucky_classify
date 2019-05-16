# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import cv2
import os
import sys

from sklearn.model_selection import train_test_split


def main():
    """
    訓練データとテストデータに分ける
    """
    folder_all_list = [
        ["data/input/gucky/", "data/input/smile/"],  # 元データのみ
        ["data/input/gucky_generate/", "data/input/smile_generate/"],
        ["data/input/gucky/", "data/input/gucky_generate_manual/", "data/input/smile/", "data/input/smile_generate_manual/"],
        ["data/input/gucky_generate/", "data/input/gucky_generate_manual/", "data/input/smile_generate/", "data/input/smile_generate_manual/"],
        ]
    
    output_filename = [
        ("train_only_raw.npy",       "test_only_raw.npy"),
        ("train_generated_by_keras.npy", "test_generated_by_keras.npy"),
        ("train_generated_manually.npy", "test_generated_manually.npy"),
        ("train_all_data.npy",           "test_all_data.npy"),
    ]
    
    for i, folder_list in enumerate(folder_all_list):
        print(folder_list)
        tmp = len(folder_list) // 2
        
        X = []
        y = []
        for j, folder in enumerate(folder_list):
            print(folder)
            filelist = os.listdir(folder)
            if j+1 <= tmp:
                true_val = 1
            else:
                true_val = 0
            
            for file in filelist:
                img = cv2.imread(folder + file)  # ndarray
                X.append(img)
                y.append(true_val)
        
        # numpy配列
        X = np.array(X)
        print("X shape: ", X.shape)
        y = np.array(y)
        print("y shape: ", y.shape)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
        train = (X_train, y_train, np.array([]))
        np.save("data/input/" + output_filename[i][0], train)
        
        test = (X_test, y_test, np.array([]))
        np.save("data/input/" + output_filename[i][1], test)


        
if __name__ == "__main__":
    main()