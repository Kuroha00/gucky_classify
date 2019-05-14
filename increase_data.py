# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import cv2
import os
import sys


from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.image import ImageDataGenerator

from utils import make_dir


# 設定できるパラメータ
# ImageDataGenerator(
#     featurewise_center=False,
#     samplewise_center=False,
#     featurewise_std_normalization=False,
#     samplewise_std_normalization=False,
#     zca_whitening=False,
#     zca_epsilon=1e-06,
#     rotation_range=0.0,
#     width_shift_range=0.0,
#     height_shift_range=0.0,
#     brightness_range=None,
#     shear_range=0.0,
#     zoom_range=0.0,
#     channel_shift_range=0.0,
#     fill_mode='nearest',
#     cval=0.0,
#     horizontal_flip=False,
#     vertical_flip=False,
#     rescale=None,
#     preprocessing_function=None,
#     data_format=None,
#     validation_split=0.0
# )


def main():
    """
    kerasによるデータを水増し
    参考 https://newtechnologylifestyle.net/keras_imagedatagenerator/
    """

    input_dir = "data/input/smile/"   # "data/input/gucky/"
    filelist = os.listdir(input_dir)
    
    output_dir = "data/input/smile_generate/"  # "data/input/gucky_generate/"
    make_dir(output_dir)
    
    for i, file in enumerate(filelist):
        img = load_img(input_dir + file)  # <class 'PIL.JpegImagePlugin.JpegImageFile'>
        x = img_to_array(img)  # ndarray, (128, 128, 3)
        x = np.expand_dims(x, axis=0) # x.shape: (1, 128, 128, 3)  テンソル
        
        # ImageDataGeneratorの生成
        datagen = ImageDataGenerator(
            width_shift_range=0.3,  # 横方向にシフト
            # height_shift_range=0.3,
        )
        
        g = datagen.flow(x, batch_size=1, save_to_dir=output_dir, save_prefix="jpg", save_format="jpg")
        for i in range(9):  # 9枚生成
            batch = g.next()
        

def test():
    """
    kerasを使わずにデータ数を増やす
    参考 https://qiita.com/bohemian916/items/9630661cd5292240f8c7
    """
    input_dir = "data/input/gucky/"   # "data/input/gucky/"
    filelist = os.listdir(input_dir)
    
    output_dir = "data/input/gucky_generate_manual/"  # "data/input/gucky_generate/"
    make_dir(output_dir)
    
    # コントラスト調整
    min_table = 50
    max_table = 205
    diff_table = max_table - min_table  # 165
    LUT_HC = np.arange(256, dtype = 'uint8' )
    LUT_LC = np.arange(256, dtype = 'uint8' )
    
    # ハイコントラストLUT作成
    for i in range(0, min_table):
        LUT_HC[i] = 0
    for i in range(min_table, max_table):
        LUT_HC[i] = 255 * (i - min_table) / diff_table
    for i in range(max_table, 255):
        LUT_HC[i] = 255
    
    # ローコントラストLUT作成
    for i in range(256):
        LUT_LC[i] = min_table + i * (diff_table) / 255
    
    # 平滑化
    average_squeare = (10,10)
    
    # ガウシアン分布によるノイズ
    mean, sigma = 0, 10

    # Salt&Pepperノイズ
    s_vs_p = 0.5
    amount = 0.004
    
    
    for i, file in enumerate(filelist):
        img = cv2.imread(input_dir + file)  # numpy配列で取得 (128, 128, 3)
        row, col, ch = img.shape
        filename = os.path.splitext(file)[0]
        
        high_cont_img = cv2.LUT(img, LUT_HC)  # ハイコントラスト
        cv2.imwrite(output_dir + filename + "_LUT_HC" + ".jpg", high_cont_img)
        
        
        low_cont_img = cv2.LUT(img, LUT_LC)  # ローコントラスト
        cv2.imwrite(output_dir + filename + "_LUT_LC" + ".jpg", low_cont_img)
        

        blur_img = cv2.blur(img, average_squeare)  # 平滑化
        cv2.imwrite(output_dir + filename + "_blur" + ".jpg", blur_img)
        
        gauss = np.random.normal(mean, sigma, (row, col, ch))  # (128, 128, 3)
        gauss = gauss.reshape(row, col, ch)  # reshapeする必要
        gauss_img = img + gauss  # ガウシアン分布のノイズ
        cv2.imwrite(output_dir + filename + "_gauss" + ".jpg", gauss_img)
        
        
        # 塩モード
        sp_img = img.copy()
        num_salt = np.ceil(amount * img.size * s_vs_p)
        coords = [np.random.randint(0, i-1 , int(num_salt)) for i in img.shape]
        sp_img[coords[:-1]] = (255,255,255)
        
        # 胡椒モード
        num_pepper = np.ceil(amount* img.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i-1 , int(num_pepper)) for i in img.shape]
        sp_img[coords[:-1]] = (0,0,0)
        cv2.imwrite(output_dir + filename + "_salt_pepper" + ".jpg", sp_img)
        
        
        # 反転
        pass
        
        # 拡大・縮小
        pass
        
        
    

    
if __name__ == "__main__":
    main()
    # test()