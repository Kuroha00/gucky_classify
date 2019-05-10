# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os
import sys

from utils import make_dir, face_recognition


def main():
    
    # folder_list = ["img_ando", "img_iwakuma", "img_nakajima", "img_nishikori", "img_ronaujinyo", "img_toda", "img_smile"]
    folder_list = ["input/gucky_miss", "input/smile_miss"]
    minNeighbors = 3
    
    for folder in folder_list:
        print(folder)
        path = os.path.join("data", folder, "")
        filelist = os.listdir(path)
        
        # if folder == "img_smile":
        if folder == "input/smile_miss":
            output_folder = "data/input/smile/"
            output_miss_folder = "data/input/smile_miss/"
        else:
            output_folder = "data/input/gucky/"
            output_miss_folder = "data/input/gucky_miss/"
        
        make_dir(output_folder)
        make_dir(output_miss_folder)
        
        for file in filelist:
            # print(file)
            
            face_recognition(path=path, file=file, output_folder=output_folder, output_miss_folder=output_miss_folder, minNeighbors=minNeighbors, minSize=(30, 30))
       
            

if __name__ == "__main__":
    main()
    # test()