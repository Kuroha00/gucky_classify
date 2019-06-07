# -*- coding: utf-8 -*-
"""
GradCamで可視化
"""
# Qiita
# https://qiita.com/haru1977/items/45269d790a0ad62604b3
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# import tensorflow as tf
import cv2
from PIL import Image
import base64
import keras
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D
from keras.optimizers import Adam
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.utils import np_utils  # One-Hot表現をするために使用
from keras.models import load_model
from keras import backend as K
from keras.preprocessing.image import array_to_img, img_to_array, load_img

from utils import make_batchdata, make_dir, push_line, face_recognition_and_reshape


def grad_cam(input_model, x, layer_name):
   '''
   Args:
      input_model: モデルオブジェクト
      x: 画像(array)
      layer_name: 畳み込み層の名前
   
   Returns:
      jetcam: 影響の大きい箇所を色付けした画像(array)
   '''
   # 前処理
   X = np.expand_dims(x, axis=0)
   X = X.astype('float32')
   X_std = X / 255.0
   
   # 予測クラスの算出
   predictions = model.predict(X_std)  # 各クラスの確率が返される
   print("predictions: ", predictions)
   class_idx = np.argmax(predictions[0])  # クラス名
   class_output = model.output[:, class_idx]
   print("class_output: ", class_output)
   
   #  勾配を取得
   conv_output = model.get_layer(layer_name).output   # layer_nameのレイヤーのアウトプット
   print("conv_output: ", conv_output)
   grads = K.gradients(class_output, conv_output)[0]  # gradients(loss, variables) で、variablesのlossに関しての勾配を返す
   print("grads: ", grads)
   
   # model.input, ??
   gradient_function = K.function([model.input], [conv_output, grads])  # model.inputを入力すると、conv_outputとgradsを出力する関数
   print("gradient function: ", gradient_function)
   
   output, grads_val = gradient_function([X_std])
   output, grads_val = output[0], grads_val[0]
   print("output: ", output)
   
   # 重みを平均化して、レイヤーのアウトプットに乗じる
   weights = np.mean(grads_val, axis=(0, 1))
   print("weights: ", weights)
   cam = np.dot(output, weights)
   
   
   # 画像化してヒートマップにして合成
   cam = cv2.resize(cam, (x.shape[0], x.shape[1]), cv2.INTER_LINEAR)
   cam = np.maximum(cam, 0)
   cam = cam / cam.max()
   
   # モノクロ画像に疑似的に色をつける
   jetcam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
   # 色をRGBに変換
   jetcam = cv2.cvtColor(jetcam, cv2.COLOR_BGR2RGB)
   # もとの画像に合成
   jetcam = (np.float32(jetcam) + x / 2)
   
   return jetcam


if __name__ == "__main__":
   
   trained_model_epoch = input("trained model epoch: ")  # 学習済みモデル
   data_set = input("data set choice(all or keras or manually or only raw): ")
   path = "./keras-model"
   model_path = os.path.join(path, data_set, "")
   
   # モデル読み込み
   model = keras.models.load_model(model_path+'model_{}.h5'.format(trained_model_epoch), compile=True)
   
   # 層の構成とかを取得できる
   print("model.summary()", model.summary())
   
   # テストデータを読み込む
   test_img = "./data/test_sample/2.jpg"
   face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt2.xml')
   x = face_recognition_and_reshape(filename=test_img, face_cascade=face_cascade)
   
   grad_cam = grad_cam(input_model=model, x=x, layer_name="activation_2")
   
   # encoded = base64.b64encode(grad_cam.tostring())
   # img = base64.b64decode(encoded)
   # img = np.frombuffer(img, dtype=np.uint8)
   # grad_cam_img = cv2.imdecode(grad_cam, cv2.IMREAD_COLOR)
   
   grad_cam_img = array_to_img(grad_cam)
   # grad_cam_img = Image.fromarray(grad_cam)
   # print("grad_cam_img: ", type(grad_cam_img))
   
   cv2.imshow("color", grad_cam)
   cv2.waitKey(0)
   cv2.destroyAllWindows()
   