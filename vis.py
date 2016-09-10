#!/usr/bin/python
#-*- coding:utf-8 -*-

"""
validate Ladder Network
"""

import os
import cv2
import time
from datetime import datetime

import numpy as np
import tensorflow as tf

import mnist_input
import model

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('dir_log', './vis_log',
                                                     """Directory where to write event logs """
                                                     """and checkpoint.""")
tf.app.flags.DEFINE_string('dir_parameter', './parameter',
                                                     """Directory where to write parameters""")
np.set_printoptions(precision=2)

def restore_model(sess):
    saver = tf.train.Saver(tf.trainable_variables())    
    ckpt = tf.train.get_checkpoint_state(FLAGS.dir_parameter)
    if ckpt and ckpt.model_checkpoint_path:
        # Restores from checkpoint
        saver.restore(sess, ckpt.model_checkpoint_path)
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
        print('No checkpoint file found')
        return None
    
    return global_step


def collect_features(network, dummy_input, idx_layer, input_image):
    # Start running operations on the Graph.
    start_layer_output = network.conv_outputs[idx_layer]
    start_layer = network.conv_layers[idx_layer]    
    N, H, W, C = start_layer_output.get_shape()
    
    dummy_feat = tf.placeholder(tf.float32, shape=(N,H,W,C))
    reconstructed = network.deconv_from_layer(idx_layer, dummy_feat)

    gpu_options = tf.GPUOptions(allow_growth=True) 
    sess = tf.Session(config=tf.ConfigProto(
        gpu_options=gpu_options, allow_soft_placement=True))
    init = tf.initialize_all_variables()        
    sess.run(init)
    global_step = restore_model(sess)

    if global_step is None:
         return

    cur_feat = sess.run(start_layer_output, feed_dict={dummy_input:input_image})     
    reconsts = []
    
    for idx_feat in range(C):
        input_feat = np.zeros(cur_feat.shape, dtype=np.float32)

        #input_feat[0,:,:,idx_feat] = cur_feat[0,:,:,idx_feat]

        pos = np.argmax(cur_feat[0,:,:,idx_feat])

        h, w = pos/int(W), pos%int(W)
        input_feat[0,h,w,idx_feat] = cur_feat[0,h,w,idx_feat]


        cur_reconst = sess.run(reconstructed,
                               feed_dict={
                                   dummy_feat:input_feat, dummy_input:input_image
                               })

        reconsts.append(cur_reconst[0])

    return reconsts


def evaluate():
    with tf.Graph().as_default() as g, tf.device("/gpu:0"):
        network = model.Network()
        dummy_input = tf.placeholder(tf.float32, shape=(1,28,28,1))

        # define forward path
        logits = network.inference(dummy_input)
        labels = tf.sigmoid(logits)

        summary_op = tf.merge_all_summaries() 
        summary_writer = tf.train.SummaryWriter(FLAGS.dir_log, g)

        original_image, input_image = mnist_input.get_image(4)
        idx_layer = 3
        feat_images = collect_features(network, dummy_input, idx_layer, input_image)

        max_value = np.max(feat_images) + 0.001
        
        for idx_feat, feat_image in enumerate(feat_images):
            path_out = os.path.join("out", "{}.png".format(idx_feat))
            normalized = (feat_image / max_value * 255).astype(int)
            cv2.imwrite(path_out, normalized)
        
                
def main(argv=None):    # pylint: disable=unused-argument
    mnist_input.init()

    if tf.gfile.Exists(FLAGS.dir_log):
        tf.gfile.DeleteRecursively(FLAGS.dir_log)
    tf.gfile.MakeDirs(FLAGS.dir_log)
    
    evaluate()

if __name__ == '__main__':
    tf.app.run()

