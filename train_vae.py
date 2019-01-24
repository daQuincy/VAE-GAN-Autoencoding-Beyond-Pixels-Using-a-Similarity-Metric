# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 21:34:35 2019

@author: YQ
"""

from vaegan import build_vae
from vaegan import batch_generator
from vaegan import train_vae_stuff
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

if not os.path.exists("vae_l_i"):
    os.mkdir("vae_l_i")

print("[INFO] Building graph...")
vae = build_vae(config.batch_size, True)
training = train_vae_stuff(vae)

d_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator")
l_restore = tf.train.Saver(var_list=d_var)

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True

sess = tf.Session(config=tf_config)
writer = tf.summary.FileWriter("vae_l", sess.graph)

sess.run(training["init"])
l_restore.restore(sess, "gan/gan")

print("[INFO] Start training...")
count = 1
for x in batch_generator(data):
    z = np.random.normal(size=(config.batch_size, config.latent_dim))
    z = z.astype(np.float32)
    fd = {vae["X"]: x, vae["is_train"]: True}

    runs = [training["global_step"], training["summ_op"]]

    sess.run(training["train_op"], feed_dict=fd)

    step, summ = sess.run(runs, feed_dict=fd)
    writer.add_summary(summ, step)

    if (step+1)%((len(data)//config.batch_size)) == 0:
        training["saver"].save(sess, "vae_l/vae-{}".format(step+1))
        samples = sess.run(vae["recon"], feed_dict={vae["is_train"]: False, vae["X"]: x})
        ori_m = imutils.build_montages([i for i in x], (64, 64), (8, 8))[0]
        montage = imutils.build_montages([i for i in samples], (64, 64), (8, 8))[0]
        m = imutils.build_montages([ori_m, montage], (512, 512), (2, 1))[0]
        cv2.imwrite("vae_l_i/iteration-{}.png".format(step+1), m)

        print("{} epoch(s) done".format(count))
        count += 1

    if (step+1)%((len(data)//config.batch_size)*config.epochs) == 0:
        training["saver"].save(sess, "vae_l/vae")
        break

sess.close()
