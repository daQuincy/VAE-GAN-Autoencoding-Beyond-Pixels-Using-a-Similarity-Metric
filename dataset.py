# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 15:19:34 2019

@author: YQ
"""

import cv2
import glob
import numpy as np
from utils.aspect_aware_resize import AspectAwareResize
import pickle
import os
from sklearn.model_selection import train_test_split

files = glob.glob("img_align_celeba/*")

imgs = []
aar = AspectAwareResize(64, 64)
count = 0
for file in files:
    f_name = os.path.split(file)[-1]
    print("[INFO] Processing {}".format(f_name))
    img = cv2.imread(file)
    img = aar.preprocess(img)
    imgs.append(img)
    
    count += 1
    if count == 50:
        break
    
imgs = np.stack(imgs)
X_train, X_test = train_test_split(imgs, test_size=200)
pickle.dump((X_train, X_test), open("celeba.p", "wb"))