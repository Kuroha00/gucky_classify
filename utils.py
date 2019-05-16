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


def face_recognition_and_reshape(filename, face_cascade, minNeighbors=3, minSize=(30,30), resize=128):
    """
    顔認識と画像サイズ変換を行って(resize, resize)の画像データを返す．
    filename: str 画像パス（ファイル名も含む）
    face_cascade: cv2.CascadeClassifierのようなもの
    minNeighbors: int   https://blog.mudatobunka.org/entry/2016/10/03/014520 参照
    minSize: tuple
    resize: int 画像のリシェイプ値
    """
    img = cv2.imread(filename)
    
    # face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt2.xml')
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    facerect = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.11,
        minNeighbors=minNeighbors,
        minSize=minSize
        )
    
    faces = face_cascade.detectMultiScale(gray)  # (x, y, w, h)
    for (x, y, w, h) in faces:
        # 顔部分取得
        roi_color = img[y:y + h, x:x + w]
        
        # roiをリサイズしてサイズを統一する
        X = cv2.resize(roi_color, (resize, resize))
    
    return X


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
    f = open("line_key.txt")
    token = f.read()
    f.close()
    token = "u8sR52QyZ9k34UCoxIpzBeTP1QT4DfTtO7S2doPJYQI"   # アクセストークン
    
    # 画像もLINEで送る job.pyと同じディレクトリにpictureディレクトリを置いている
    folderpath = "./dog_picture/"
    pic_list = os.listdir(folderpath)
    n = np.random.randint(0, len(pic_list))
    
    headers = {"Authorization" : "Bearer " + token}
    # message = filename + " write"
    payload = {"message": message}
    files = {"imageFile": open(folderpath + pic_list[n], "rb")}
    
    r = requests.post(url, headers=headers, params=payload, files=files)