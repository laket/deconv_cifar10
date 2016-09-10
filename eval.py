#!/usr/bin/python
#-*- coding:utf-8 -*-

"""
validate Ladder Network
"""

import os
import time
from datetime import datetime

import numpy as np
import tensorflow as tf

import mnist_input
import model

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('dir_log', './validation_log',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('dir_parameter', './parameter',
                           """Directory where to write parameters""")

tf.app.flags.DEFINE_integer('eval_interval_secs', 30,
                            """How often to run the eval.""")


def restore_model(saver, sess):
  ckpt = tf.train.get_checkpoint_state(FLAGS.dir_parameter)
  if ckpt and ckpt.model_checkpoint_path:
    # Restores from checkpoint
    saver.restore(sess, ckpt.model_checkpoint_path)
    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
  else:
    print('No checkpoint file found')
    return None
  
  return global_step

def eval_once(summary_writer, top_k_op, entropy):
  saver = tf.train.Saver(tf.trainable_variables())
  # Build an initialization operation to run below.
  init = tf.initialize_all_variables()

  # Start running operations on the Graph.

  gpu_options = tf.GPUOptions(allow_growth=True) 
  sess = tf.Session(config=tf.ConfigProto(
    gpu_options=gpu_options, allow_soft_placement=True))
  sess.run(init)
  global_step = restore_model(saver, sess)

  if global_step is None:
    return

  # Start the queue runners.
  coord = tf.train.Coordinator()    
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)

  true_count = 0  # Counts the number of correct predictions.
  total_sample_count = mnist_input.VALIDATION_SIZE
  step = 0
  num_iter = total_sample_count / FLAGS.batch_size
  entropies = []
    
  for i in range(num_iter):
    predictions, value_entropy = sess.run([top_k_op, entropy])
    true_count += np.sum(predictions)
    step += 1
    entropies.append(value_entropy)

  # Compute precision @ 1.
  precision = true_count / float(total_sample_count)
  mean_entropy = float(np.mean(entropies))
  
  print('step %d precision @ 1 = %.3f entropy = %.3f' % (int(global_step), precision, mean_entropy))

  summary = tf.Summary()
  summary.value.add(tag='Precision @ 1', simple_value=precision)
  summary.value.add(tag='entropy', simple_value=mean_entropy)  
  summary_writer.add_summary(summary, global_step)            


  coord.request_stop()
  coord.join(threads, stop_grace_period_secs=10)
    

def evaluate():
  with tf.Graph().as_default() as g, tf.device("/gpu:0"):
    FLAGS.batch_size = 100
    images, labels = mnist_input.validate_input()
    label_vector = tf.one_hot(labels, 10, dtype=tf.float32)

    network = model.Network()    
    logits = network.inference(images)
    top_k_op = tf.nn.in_top_k(logits, labels, 1)
    entropy, loss = model.get_loss(label_vector, logits)    

    summary_writer = tf.train.SummaryWriter(FLAGS.dir_log, g)

    while True:
      eval_once(summary_writer, top_k_op, entropy)
      time.sleep(FLAGS.eval_interval_secs)

        
def main(argv=None):  # pylint: disable=unused-argument
  mnist_input.init()
  
  if tf.gfile.Exists(FLAGS.dir_log):
    tf.gfile.DeleteRecursively(FLAGS.dir_log)
  tf.gfile.MakeDirs(FLAGS.dir_log)
  
  evaluate()

if __name__ == '__main__':
  tf.app.run()

