# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 20:20:49 2019

@author: YQ
"""

from vaegan import build_vaegan
from vaegan import batch_generator
from vaegan import train_stuff
import os
import config
import tensorflow as tf
import pickle
import imutils
import cv2
import numpy as np

tf.reset_default_graph()

print("[INFO] Loading data...")
data, _ = pickle.load(open("celeba.p", "rb"))

if not os.path.exists("vaegan_i"):
    os.mkdir("vaegan_i")

print("[INFO] Building graph...")
gan = build_vaegan(config.batch_size)
training = train_stuff(gan)

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True

sess = tf.Session(config=tf_config)
writer = tf.summary.FileWriter("vaegan", sess.graph)

sess.run(training["init"])

print("[INFO] Start training...")
count = 1
for x in batch_generator(data):
    z = np.random.normal(size=(config.batch_size, config.latent_dim))
    z = z.astype(np.float32)
    fd = {gan["X"]: x, gan["is_train"]: True, gan["Z"]: z}

    runs = [training["global_step"], training["summ_op"]]

    sess.run(training["dis_op"], feed_dict=fd)
    sess.run(training["dec_op"], feed_dict=fd)
    sess.run(training["enc_op"], feed_dict=fd)

    step, summ = sess.run(runs, feed_dict=fd)
    writer.add_summary(summ, step)

    if (step+1)%((len(data)//config.batch_size)) == 0:
        training["saver"].save(sess, "vaegan/vaegan-{}".format(step+1))
        samples = sess.run(gan["recon"], feed_dict={gan["is_train"]: False, gan["X"]: x})
        montage = imutils.build_montages([i for i in samples], (64, 64), (8, 8))[0]
        ori_m = imutils.build_montages([i for i in x], (64, 64), (8, 8))[0]
        m = imutils.build_montages([ori_m, montage], (512, 512), (2, 1))[0]
        cv2.imwrite("vaegan_i/iteration-{}.png".format(step+1), m)

        print("{} epoch(s) done".format(count))
        count += 1

    if (step+1)%((len(data)//config.batch_size)*config.epochs) == 0:
        training["saver"].save(sess, "vaegan/vaegan")
        break

sess.close()
