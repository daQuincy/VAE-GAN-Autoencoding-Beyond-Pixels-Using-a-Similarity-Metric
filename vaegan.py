# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 23:39:48 2019

@author: YQ
"""

import tensorflow as tf
import numpy as np
import config

init = tf.truncated_normal_initializer
act = tf.nn.relu

def conv2d(x, output_dim, k_size=5, stride=2, stddev=0.02, name="conv2d"):
    conv = tf.layers.conv2d(x, output_dim, kernel_size=k_size, strides=[stride, stride],
                            padding="SAME", kernel_initializer=init(stddev=0.02),
                            name=name)

    return conv

def deconv2d(x, output_dim, k_size=5, stride=2, name="deconv2d"):
    deconv = tf.layers.conv2d_transpose(x, output_dim, kernel_size=k_size, strides=[stride, stride],
                                        padding="SAME", kernel_initializer=init(stddev=0.02),
                                        name=name)

    return deconv

def fc(x, output_dim, name="dense"):
    dense = tf.layers.dense(x, output_dim, kernel_initializer=init(stddev=0.02),
                            name=name)

    return dense

def batch_norm(x, training, name="bn", reuse=tf.AUTO_REUSE):
    bn = tf.contrib.layers.batch_norm(
            x, is_training=training, epsilon=1e-5, decay=0.9, reuse=reuse, scope=name,
            updates_collections=None)

    return bn

def KL(mu, sigma):
    loss = 0.5 * tf.reduce_sum(
            (tf.square(mu) + tf.square(sigma) - tf.log(tf.square(sigma)+1e-8) - 1),
            axis=-1) / config.latent_dim

    return loss

def decoder(x, is_train, reuse=False):
    with tf.variable_scope("decoder") as scope:
        if reuse:
            scope.reuse_variables()

        y = fc(x, 8*8*256, "dec_fc1")
        y = batch_norm(tf.reshape(y, shape=(-1, 8, 8, 256)), is_train, "dec_bn1", reuse)
        y = act(batch_norm(deconv2d(y, 256, name="dec_c1"), is_train, "dec_bn2", reuse))
        y = act(batch_norm(deconv2d(y, 128, name="dec_c2"), is_train, "dec_bn3", reuse))
        y = act(batch_norm(deconv2d(y, 32, name="dec_c3", stride=1), is_train, "dec_bn4", reuse))
        y = deconv2d(y, 3, name="dec_c4")
        y = tf.nn.tanh(y)

        return y

def discriminator(x, is_train, reuse=False):
    with tf.variable_scope("discriminator") as scope:
        if reuse:
            scope.reuse_variables()

        y = act(batch_norm(conv2d(x, 32, name="dis_c1"), is_train, "dis_bn1", reuse))
        y = act(batch_norm(conv2d(y, 128, name="dis_c2"), is_train, "dis_bn2", reuse))
        y = act(batch_norm(conv2d(y, 256, name="dis_c3"), is_train, "dis_bn3", reuse))
        y_ = conv2d(y, 256, name="dis_c4")
        y = act(batch_norm(y_, is_train, "dis_bn4", reuse))

        y = tf.layers.flatten(y)
        y = fc(y, 512, "dis_l")
        y = act(y)
        y = fc(y, 1, "dis_fc")
        y = tf.nn.sigmoid(y)

        return y, y_

def encoder(x, is_train):
    with tf.variable_scope("encoder"):
        y = act(batch_norm(conv2d(x, 64, name="enc_c1"), is_train, "enc_bn1"))
        y = act(batch_norm(conv2d(y, 128, name="enc_c2"), is_train, "enc_bn2"))
        y = act(batch_norm(conv2d(y, 256, name="enc_c3"), is_train, "enc_bn3"))
        y = tf.layers.flatten(y)
        y = act(batch_norm(fc(y, 2048, name="enc_fc"), is_train, "enc_bn4"))

        mean = fc(y, config.latent_dim, name="enc_mean")
        sigma = fc(y, config.latent_dim, name="enc_sigma")

        return mean, sigma

def build_gan(batch_size):
    X_ = tf.placeholder(tf.float32, [batch_size, 64, 64 ,3], name="X")
    X = (X_ / 127.5) - 1
    z_p = tf.placeholder(tf.float32, shape=(batch_size, config.latent_dim))
    is_train = tf.placeholder(tf.bool)

    x_p = decoder(z_p, is_train, False)

    D_real, _ = discriminator(X, reuse=False, is_train=is_train)
    D_fake, _ = discriminator(x_p, reuse=True, is_train=is_train)

    with tf.variable_scope("loss"):
        dec_loss = -tf.reduce_mean(tf.log((D_fake*0.45)+1e-8))
        dis_loss = -tf.reduce_mean(tf.log((D_real*0.9)+1e-8) + tf.log(1.-D_fake+1e-8))

    with tf.variable_scope("summ_loss"):
        summaries = [tf.summary.scalar("dis_loss", dis_loss),
                     tf.summary.scalar("dec_loss", dec_loss)]

    with tf.variable_scope("accuracies"):
        summaries += [tf.summary.scalar("D_real", tf.reduce_mean(D_real)),
                      tf.summary.scalar("D_fake", tf.reduce_mean(D_fake))]

    gan = {"X": X_, "Z": z_p, "dec_loss": dec_loss, "dis_loss": dis_loss, "summaries": summaries,
           "generate": (x_p+1)*127.5, "is_train": is_train}

    return gan

def build_vae(batch_size, with_l=False):
    X_ = tf.placeholder(tf.float32, [batch_size, 64, 64 ,3], name="X")
    X = (X_ / 127.5) - 1
    is_train = tf.placeholder(tf.bool)

    z_mean, z_log_sigma = encoder(X, is_train)

    eps = tf.random_normal([batch_size, config.latent_dim])
    z = z_mean + tf.exp(z_log_sigma)*eps

    x_tilde = decoder(z, is_train, False)
    
    if with_l:
        _, l_x = discriminator(X, is_train, False)
        _, l_tilde = discriminator(x_tilde, is_train, True)

    with tf.variable_scope("loss"):
        kl_loss = tf.reduce_mean(KL(z_mean, tf.exp(z_log_sigma)))
        mse_loss = tf.reduce_mean(tf.reduce_sum(tf.square(X-x_tilde), axis=[1,2,3]))

        loss = kl_loss + mse_loss
        
        if with_l:
            ll_loss = tf.reduce_mean(tf.reduce_sum(tf.square(l_tilde-l_x), axis=[1,2,3])) / (4*4*256)
            loss += ll_loss

    with tf.variable_scope("summ_loss"):
        summaries = [tf.summary.scalar("kl_loss", kl_loss),
                     tf.summary.scalar("mse_loss", mse_loss),
                     tf.summary.scalar("loss", loss)]
        
        if with_l:
            summaries.append(tf.summary.scalar("ll_loss", ll_loss))

    vae = {"X": X_, "is_train": is_train, "recon": (x_tilde+1)*127.5, "summaries": summaries,
           "kl_loss": kl_loss, "mse_loss": mse_loss, "loss": loss, "Z": z}

    return vae

def build_vaegan(batch_size):
    X_ = tf.placeholder(tf.float32, [batch_size, 64, 64 ,3], name="X")
    X = (X_ / 127.5) - 1
    is_train = tf.placeholder(tf.bool)

    z_mean, z_log_sigma = encoder(X, is_train)
    eps = tf.random_normal([batch_size, config.latent_dim])
    z_tilde = z_mean + tf.exp(z_log_sigma)*eps
    x_tilde = decoder(z_tilde, is_train, False)

    z_p = tf.placeholder(tf.float32, shape=(batch_size, config.latent_dim))
    x_p = decoder(z_p, is_train, True)

    D_real, l_real = discriminator(X, reuse=False, is_train=is_train)
    D_fake, l_fake = discriminator(x_p, reuse=True, is_train=is_train)
    D_tilde, l_tilde = discriminator(x_tilde, reuse=True, is_train=is_train)

    with tf.variable_scope("loss"):
        ll_loss = tf.reduce_mean(tf.reduce_sum(tf.square(l_tilde-l_real), axis=[1,2,3]))
        kl_loss = tf.reduce_mean(KL(z_mean, tf.exp(z_log_sigma)))

        enc_loss = kl_loss + (ll_loss / (4*4*256))
        dis_loss = -tf.reduce_mean(tf.log(D_real+1e-8) + tf.log(1.-D_fake+1e-8) + tf.log(1.-D_tilde+1e-8))
        dec_loss = -tf.reduce_mean(tf.log(D_tilde+1e-8) + tf.log(D_fake+1e-8)) + 0.2*(ll_loss / (4*4*256))


    with tf.variable_scope("summ_loss"):
        summaries = [tf.summary.scalar("dis_loss", dis_loss),
                     tf.summary.scalar("dec_loss", dec_loss),
                     tf.summary.scalar("enc_loss", enc_loss),
                     tf.summary.scalar("kl_loss", kl_loss),
                     tf.summary.scalar("ll_loss", ll_loss)]

    with tf.variable_scope("accuracies"):
        summaries += [tf.summary.scalar("D_real", tf.reduce_mean(D_real)),
                      tf.summary.scalar("D_fake", tf.reduce_mean(D_fake)),
                      tf.summary.scalar("D_tilde", tf.reduce_mean(D_tilde))]

    gan = {"X": X_, "dec_loss": dec_loss, "dis_loss": dis_loss, "summaries": summaries,
           "recon": (x_tilde+1)*127.5, "is_train": is_train, "enc_loss": enc_loss,
           "Z": z_p, "generate": (x_p+1)*127.5, "Z_t": z_tilde}

    return gan


def train_stuff(model):
    global_step = tf.train.create_global_step()

    learning_rate = tf.train.exponential_decay(config.learning_rate, global_step=global_step,
                                                  decay_steps=10000, decay_rate=0.98)

    train_vars = tf.trainable_variables()
    dec_vars = [var for var in train_vars if "decoder" in var.name]
    dis_vars = [var for var in train_vars if "discriminator" in var.name]
    enc_vars = [var for var in train_vars if "encoder" in var.name]
    dec_opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
    dis_opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
    enc_opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
    dec_grad = dec_opt.compute_gradients(model["dec_loss"], var_list=dec_vars)
    dis_grad = dis_opt.compute_gradients(model["dis_loss"], var_list=dis_vars)
    enc_grad = enc_opt.compute_gradients(model["enc_loss"], var_list=enc_vars)
    dec_op = dec_opt.apply_gradients(dec_grad)
    dis_op = dis_opt.apply_gradients(dis_grad)
    enc_op = enc_opt.apply_gradients(enc_grad, global_step=global_step)

    summaries = [tf.summary.scalar("lr", learning_rate)] + model["summaries"]
    summ_op = tf.summary.merge(summaries)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    stuff = {"global_step": global_step, "dec_op": dec_op, "dis_op": dis_op,
             "summ_op": summ_op, "init": init, "saver": saver, "enc_op": enc_op}

    return stuff

def train_vae_stuff(model):
    global_step = tf.train.create_global_step()

    learning_rate = tf.train.exponential_decay(config.learning_rate, global_step=global_step,
                                                  decay_steps=10000, decay_rate=0.98)
    enc_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="encoder")
    dec_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="decoder")
    train_var = enc_var + dec_var
    
    opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
    train_op = opt.minimize(model["loss"], global_step=global_step, var_list=train_var)
    summaries = [tf.summary.scalar("lr", learning_rate)] + model["summaries"]
    summ_op = tf.summary.merge(summaries)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    stuff = {"train_op": train_op, "summ_op": summ_op, "global_step": global_step, "init": init,
             "saver": saver}

    return stuff


def train_gan_stuff(model):
    global_step = tf.train.create_global_step()

    learning_rate = tf.train.exponential_decay(config.learning_rate, global_step=global_step,
                                                  decay_steps=10000, decay_rate=0.98)

    train_vars = tf.trainable_variables()
    dec_vars = [var for var in train_vars if "decoder" in var.name]
    dis_vars = [var for var in train_vars if "discriminator" in var.name]
    dec_opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
    dis_opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
    dec_grad = dec_opt.compute_gradients(model["dec_loss"], var_list=dec_vars)
    dis_grad = dis_opt.compute_gradients(model["dis_loss"], var_list=dis_vars)
    dec_op = dec_opt.apply_gradients(dec_grad)
    dis_op = dis_opt.apply_gradients(dis_grad, global_step=global_step)

    summaries = [tf.summary.scalar("lr", learning_rate)] + model["summaries"]
    summ_op = tf.summary.merge(summaries)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    stuff = {"global_step": global_step, "dec_op": dec_op, "dis_op": dis_op,
             "summ_op": summ_op, "init": init, "saver": saver}

    return stuff


def batch_generator(data):
    while True:
        idxs = np.random.permutation(len(data))
        for i in range(0, len(data)):
            up = i*config.batch_size
            down = int((i*config.batch_size) + config.batch_size)
            x = data[idxs[up:down]]

            if len(x) == config.batch_size:
                yield x
            else:
                break

if __name__ == "__main__":
    tf.reset_default_graph()
    a = build_vae(64)
