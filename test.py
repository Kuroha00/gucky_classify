# -*- coding: utf-8 -*-
"""
重みパラメータとかを可視化
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

import tensorflow as tf
from makeCNNclass import CNN


def main():
    previous_epoch_num = 150
    data_set = "all"
    model = CNN()
    model.load(previous_epoch_num=previous_epoch_num, data_set=data_set)
    # print(model.conv_h1)
    print(model.layers)
    

if __name__ == "__main__":
    main()

