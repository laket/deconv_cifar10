#!/usr/bin/python

import os
import numpy as np
import tensorflow as tf
import mnist_input
import model

tf.app.flags.DEFINE_string('dir_log', './log',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('dir_parameter', './parameter',
                           """Directory where to write parameters""")

FLAGS = tf.app.flags.FLAGS


def get_opt(loss, global_step):
    lr = tf.train.exponential_decay(0.1,
                                    global_step,
                                    5000,
                                    0.1,
                                    staircase=True)
    
    #opt = tf.train.AdamOptimizer(0.01)
    opt = tf.train.MomentumOptimizer(0.01, momentum=0.95)
    opt_op = opt.minimize(loss, global_step=global_step)

    tf.scalar_summary("lr", lr)

    return lr, opt_op
    

def train():
    global_step = tf.Variable(0, trainable=False)
    
    image, label = mnist_input.train_input()

    network = model.Network()    
    logits = network.inference(image, is_train=True)

    for var in tf.trainable_variables():
        tf.histogram_summary(var.op.name, var)
    
    entropy, loss = model.get_loss(label, logits)
    
    lr, opt = get_opt(loss, global_step)

    saver = tf.train.Saver(tf.trainable_variables())
    summary_op = tf.merge_all_summaries()
    
    gpu_options = tf.GPUOptions(allow_growth=True) 
    
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        init = tf.initialize_all_variables()
        sess.run(init)
        summary_writer = tf.train.SummaryWriter("log", sess.graph)        
        
        tf.train.start_queue_runners(sess=sess)

        for num_iter in range(1,1000000):
            value_entropy, value_loss, value_lr, _ = sess.run([entropy, loss, lr, opt])

            if num_iter % 100 == 0:
                print "lr = {}  entropy = {} loss = {}".format(value_lr, value_entropy, value_loss)
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, num_iter)

                
            if num_iter % 1000 == 0:

                checkpoint_path = os.path.join(FLAGS.dir_parameter, 'model.ckpt')                
                saver.save(sess, checkpoint_path,global_step=num_iter)
                

def main(argv=None):  # pylint: disable=unused-argument
    if tf.gfile.Exists(FLAGS.dir_log):
        tf.gfile.DeleteRecursively(FLAGS.dir_log)
    tf.gfile.MakeDirs(FLAGS.dir_log)

    if tf.gfile.Exists(FLAGS.dir_parameter):
        tf.gfile.DeleteRecursively(FLAGS.dir_parameter)
    tf.gfile.MakeDirs(FLAGS.dir_parameter)
    
    mnist_input.init()
    train()

if __name__ == '__main__':
    tf.app.run()

    
