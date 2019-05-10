# -*- coding: utf-8 -*-
import http.client
import json
import re
import requests
import math
import pickle
import urllib
import hashlib
import sha3
import os
import sys

from utils import make_dir


def make_correspondence_table(correspondence_table, original_url, hashed_url):
    """
    Create reference table of hash value and original URL.
    """
    correspondence_table[original_url] = hashed_url


def make_img_path(save_dir_path, url):
    """
    Hash the image url and create the path
    
    Args:
        save_dir_path (str): Path to save image dir.
        url (str): An url of image.
    
    Returns:
        Path of hashed image URL.
    """
    save_img_path = os.path.join(save_dir_path, 'imgs')
    make_dir(save_img_path)
    
    file_extension = os.path.splitext(url)[-1]  # ファイルの拡張子を取得 ; splitext.py -> ('splitext', '.py')
    if file_extension.lower() in ('.jpg', '.jpeg', '.png', '.bmp'):  # 変更:gifを除いた
        encoded_url = url.encode('utf-8') # required encoding for hashed
        hashed_url = hashlib.sha3_256(encoded_url).hexdigest()
        full_path = os.path.join(save_img_path, hashed_url + file_extension.lower())
        
        make_correspondence_table(correspondence_table, url, hashed_url)
        
        return full_path
    else:
        raise ValueError('Not applicable file extension')


def download_image(url, timeout=10):
    """
    画像URLから画像を引っ張ってくる
    """
    response = requests.get(url, allow_redirects=True, timeout=timeout)
    if response.status_code != 200:
        error = Exception("HTTP status: " + response.status_code)
        raise error
    
    content_type = response.headers["content-type"]
    if 'image' not in content_type:
        error = Exception("Content-Type: " + content_type)
        raise error
    
    return response.content


def save_image(filename, image):
    with open(filename, "wb") as fout:
        fout.write(image)


if __name__ == "__main__":
    save_dir_path = "data/"
    term = sys.argv[1]
    
    make_dir(save_dir_path)
    num_imgs_required = 10000 # Number of images you want. The number to be divisible by 'num_imgs_per_transaction'
    num_imgs_per_transaction = 150   # default 30, Max 150  1トランザクションで取得できる画像数
    offset_count = math.floor(num_imgs_required / num_imgs_per_transaction)
    
    url_list = []
    correspondence_table = {}
    
    headers = {
        # Request headers
        'Content-Type': 'multipart/form-data',
        'Ocp-Apim-Subscription-Key': 'azure subscription key',
    }
    
    url = "https://api.cognitive.microsoft.com/bing/v7.0/images/search"
    for offset in range(offset_count):
        print("offset: ", offset)
        
        params = {
            "q": term,
            "mkt":"ja-JP",
            "count": num_imgs_per_transaction,
            "offset": offset * num_imgs_per_transaction,
            "imageType":"Photo", 
            "color":"ColorOnly",
        }
        
        
        try:
            # conn = http.client.HTTPSConnection('api.cognitive.microsoft.com')
            # conn.request("POST", "/bing/v7.0/images/search?%s" % params, "{body}", headers)
            # response = conn.getresponse()
            # data = response.read()
            
            search = requests.get(url, headers=headers, params=params)
            search.raise_for_status()
            data = search.json()
            
            save_res_path = os.path.join(save_dir_path, 'pickle_files')
            make_dir(save_res_path)
            with open(os.path.join(save_res_path, '{}.pickle'.format(offset)), mode='wb') as f:
                pickle.dump(data, f)
            
            # conn.close()
        
        except Exception as err:
            print("[Errno {0}] {1}".format(err.errno, err.strerror))
        
        # tryで例外が発生せずに実行されたときにのみelse内が実行される
        else:
            # decode_res = data.decode('utf-8')
            # data = json.loads(decode_res)
            # pattern = r"&r=(http.+)&p="   # extract an URL of image
            
            for values in data['value']:
                unquoted_url = urllib.parse.unquote(values['contentUrl'])
                url_list.append(unquoted_url)
                
                # img_url = re.search(pattern, unquoted_url)
                # if img_url:
                #     url_list.append(img_url.group(1))
    
    
    for url in url_list:
        try:
            img_path = make_img_path(save_dir_path, url)
            image = download_image(url)
            save_image(img_path, image)
            print('saved image... {}'.format(url))
        except KeyboardInterrupt:
            break
        except Exception as err:
            print("%s" % (err))
    
    correspondence_table_path = os.path.join(save_dir_path, 'corr_table')
    make_dir(correspondence_table_path)
    
    with open(os.path.join(correspondence_table_path, 'corr_table.json'), mode='w') as f:
        json.dump(correspondence_table, f)