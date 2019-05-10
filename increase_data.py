# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
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
    """

    
if __name__ == "__main__":
    main()