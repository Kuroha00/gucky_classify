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


def Grad_Cam_plus_plus(input_model, x, layer_name):
   '''
   Zouさんのプログラムを拝借
   
   Parameters:
      input_model: 2D-CNNモデル
      x: 画像(array)
      layer_name: 畳み込み層の名前
      (row, col): 画像のサイズ
   Returns:
      jetcam: 影響の大きい箇所を色付けした画像(array)
   '''
   
   model = input_model

   #前処理
   X = x
   X = X.astype('float32')
   preprocessed_input = X

   #予測クラスの算出
   predictions = model.predict(preprocessed_input)
   class_idx = np.argmax(predictions[0]) #回帰モデルはここ要注意

   #使用する重みの抽出、高階微分の計算
   class_output = model.layers[-1].output
   #print(class_output)
   #print(layer_name)
   #print(model.get_layer(layer_name))
   conv_output = model.get_layer(layer_name).output
   grads = K.gradients(class_output, conv_output)[0]
   #first_derivative：１階微分
   first_derivative = K.exp(class_output)[0][class_idx] * grads
   #second_derivative：２階微分
   second_derivative = K.exp(class_output)[0][class_idx] * grads * grads
   #third_derivative：３階微分
   third_derivative = K.exp(class_output)[0][class_idx] * grads * grads * grads
   
   #関数の定義
   gradient_function = K.function([model.input], [conv_output, first_derivative, second_derivative, third_derivative])  # model.inputを入力すると、conv_outputとgradsを出力する関数
   
   
   conv_output, conv_first_grad, conv_second_grad, conv_third_grad = gradient_function([preprocessed_input])
   conv_output, conv_first_grad, conv_second_grad, conv_third_grad = conv_output[0], conv_first_grad[0], conv_second_grad[0], conv_third_grad[0]
   #print(conv_output, conv_first_grad, conv_second_grad, conv_third_grad)
   
   #alphaを求める
   global_sum = np.sum(conv_output.reshape((-1, conv_first_grad.shape[2])), axis=0)
   alpha_num = conv_second_grad
   alpha_denom = conv_second_grad*2.0 + conv_third_grad*global_sum.reshape((1,1,conv_first_grad.shape[2]))
   alpha_denom = np.where(alpha_denom!=0.0, alpha_denom, np.ones(alpha_denom.shape))
   alphas = alpha_num / alpha_denom

   #alphaの正規化
   alpha_normalization_constant = np.sum(np.sum(alphas, axis = 0), axis = 0)
   alpha_normalization_constant_processed = np.where(alpha_normalization_constant != 0.0, alpha_normalization_constant, np.ones(alpha_normalization_constant.shape))
   alphas /= alpha_normalization_constant_processed.reshape((1,1,conv_first_grad.shape[2]))

   #wの計算
   weights = np.maximum(conv_first_grad, 0.0)
   deep_linearization_weights = np.sum((weights * alphas).reshape((-1, conv_first_grad.shape[2])))

   #Lの計算
   grad_CAM_map = np.sum(deep_linearization_weights * conv_output, axis=2)
   grad_CAM_map = np.maximum(grad_CAM_map, 0)
   grad_CAM_map = grad_CAM_map / np.max(grad_CAM_map)
   #print(np.sum(grad_CAM_map))
   #print(grad_CAM_map)
   
   #grad_CAM_map = cv2.resize(grad_CAM_map, (row, col), cv2.INTER_LINEAR)
   #jetcam = cv2.applyColorMap(np.uint8(255 * grad_CAM_map), cv2.COLORMAP_JET)  # モノクロ画像に疑似的に色をつける
   #jetcam = (np.float32(jetcam) + x / 2)   # もとの画像に合成
   
   return grad_CAM_map, class_idx


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
   predictions = model.predict(X_std) # 各クラスの確率が返される 2次元配列
   print("predictions: ", predictions)
   class_idx = np.argmax(predictions[0])  # クラス名
   class_output = model.output[:, class_idx]
   # class_output = model.layers[-1].output
   print("class_output: ", class_output)
   
   #  勾配を取得
   conv_output = model.get_layer(layer_name).output   # layer_nameのレイヤーのアウトプット
   print("conv_output: ", conv_output)
   # gradients(loss, variables) で、variablesのlossに関しての勾配を返す
   grads = K.gradients(class_output, conv_output)[0]
   print("grads: ", grads)
   
   # model.inputを入力すると，conv_outputとgradsを出力する関数
   gradient_function = K.function([model.input], [conv_output, grads])  # model.inputを入力すると、conv_outputとgradsを出力する関数
   print("gradient function: ", gradient_function)
   
   output, grads_val = gradient_function([X_std])  # データを入力し，その出力とgradを得る
   print("output: ", output.shape)
   
   output, grads_val = output[0], grads_val[0]
   print("output: ", output.shape)
   print("grads_val: ", grads_val.shape)
   
   # outputを可視化したい
   # img = array_to_img(output[:,:,])
   # cv2.imshow(img)
   # cv2.waitKey(0)
   # cv2.destroyAllWindows()
   
   # 重みを平均化して、レイヤーのアウトプットに乗じる
   weights = np.mean(grads_val, axis=(0, 1))
   print("weights: ", weights.shape)
   cam = np.dot(output, weights)  # 出力と重みを掛ける
   print("cam: ", cam.shape)  # 最後の層だと(11, 11)
   
   # 画像化してヒートマップにして合成
   cam = cv2.resize(cam, (x.shape[0], x.shape[1]), cv2.INTER_LINEAR)
   cam = np.maximum(cam, 0)
   cam = cam / np.max(cam)
   
   # # モノクロ画像に疑似的に色をつける
   # jetcam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
   # # 色をRGBに変換
   # jetcam = cv2.cvtColor(jetcam, cv2.COLOR_BGR2RGB)
   # # もとの画像に合成
   # jetcam = (np.float32(jetcam) + x / 2)
   
   return cam, class_idx


if __name__ == "__main__":
   
   trained_model_epoch = input("trained model epoch: ")  # 学習済みモデル
   data_set = input("data set choice(all or keras or manually or only raw): ")
   img_num = input("input image number: ")
   path = "./keras-model"
   model_path = os.path.join(path, data_set, "")
   
   # モデル読み込み
   model = load_model(model_path+'model_3conv_{}.h5'.format(trained_model_epoch), compile=True)
   
   conv_layers = []
   for layer in model.layers:
      if 'conv' in layer.name:
         conv_layers.append(layer.name)
   
   # 層の構成とかを取得できる
   # print("model.summary()", model.summary())
   
   # テストデータを読み込む
   test_img = "./data/test_sample/{}.jpg".format(img_num)
   face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt2.xml')
   x = face_recognition_and_reshape(filename=test_img, face_cascade=face_cascade)
   
   # grad_cam = grad_cam(input_model=model, x=x, layer_name="activation_2")
   
   # grad_cam_img = array_to_img(grad_cam)
   # # grad_cam_img = Image.fromarray(grad_cam)
   # # print("grad_cam_img: ", type(grad_cam_img))
   
   # cv2.imshow("color", grad_cam)
   # cv2.waitKey(0)
   # cv2.destroyAllWindows()
   
   for target_layer in conv_layers:
      img_value = np.nan
      count = 0
      while np.isnan(img_value) and count <= 20:
         grad_CAM_map, pre_class = grad_cam(model, x, target_layer)
         img_value = np.sum(grad_CAM_map, axis=None)
         print(target_layer, img_value)
         count = count + 1
      
      # print('----- true class==', np.argmax(y_test[index]), 'prediction class==', pre_class, '-----')
      print("prediction class: ", pre_class)
      
      #ヒートマップを描く
      grad_CAM_map = cv2.resize(grad_CAM_map, (x.shape[0], x.shape[1]), cv2.INTER_LINEAR)
      # モノクロ画像に疑似的に色をつける
      jetcam = cv2.applyColorMap(np.uint8(255 * grad_CAM_map), cv2.COLORMAP_JET)
      img_Gplusplusname = './GCAM++_%s.jpg'%target_layer
      # cv2.imwrite(img_Gplusplusname, jetcam)
      print('----- layer', target_layer, 'visualization -----')
      plt.clf()
      # plt.figure()
      plt.imshow(cv2.cvtColor(jetcam, cv2.COLOR_BGR2RGB))
      plt.show()
   