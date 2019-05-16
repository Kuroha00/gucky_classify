# -*- coding: utf-8 -*-
import os
import sys
import requests
import numpy as np

import cv2


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def face_recognition(path, file, output_folder, output_miss_folder, minNeighbors=3, minSize=(30,30)):
    # 識別器
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt2.xml')
    
    filename, ext = os.path.splitext(file)
    
    if ext==".gif": # gifだと読み込めない
        return None
    
    img = cv2.imread(path + file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    facerect = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.11,
        minNeighbors=minNeighbors,
        minSize=minSize
        )
    
    faces = face_cascade.detectMultiScale(gray)  # (x, y, w, h)
    
    # 顔検出成功
    if len(faces)==1:
        for (x, y, w, h) in faces:
            print(file)
            
            # 顔部分取得
            roi_color = img[y:y + h, x:x + w]
            # roiをリサイズしてサイズを統一する
            resize = 128
            roi_resize = cv2.resize(roi_color, (resize, resize))
            
            # 保存
            cv2.imwrite(output_folder + filename + ".jpg", roi_resize)
            
            # 元の画像消去
            # os.remove(path+file)
    
    # 顔検出失敗
    else:
        # print("len: {}, {}".format(len(faces), file))
        cv2.imwrite(output_miss_folder + file, img)
        
        # os.remove(path + file)


def make_batchdata(X, y, batch_size=64, shuffle=True, random_seed=None):
    """
    バッチサイズのデータを渡す
    yieldで実装
    """
    idx = np.arange(y.shape[0])
    if shuffle:
        rng = np.random.RandomState(random_seed)
        rng.shuffle(idx)
        X = X[idx]
        y = y[idx]
    
    for i in range(0, X.shape[0], batch_size):
        yield( X[i:i+batch_size, :], y[i:i+batch_size] )  # 呼ばれるごとに


def push_line(message):
    """
    LINE通知用関数
    filename: str型 LINEへの通知の際のメッセージ内容に書き込んだファイル名を入れている
    """
    url = "https://notify-api.line.me/api/notify"   # LINE notify url
    token = "access token"   # アクセストークン
    
    # 画像もLINEで送る job.pyと同じディレクトリにpictureディレクトリを置いている
    folderpath = "./dog_picture/"
    pic_list = os.listdir(folderpath)
    n = np.random.randint(0, len(pic_list))
    
    headers = {"Authorization" : "Bearer " + token}
    # message = filename + " write"
    payload = {"message": message}
    files = {"imageFile": open(folderpath + pic_list[n], "rb")}
    
    r = requests.post(url, headers=headers, params=payload, files=files)