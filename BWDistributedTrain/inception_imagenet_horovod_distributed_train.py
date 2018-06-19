import socket
import os
import sys
import time
import numpy as np
from datetime import datetime as dt
import random
from itertools import groupby
from itertools import chain

path_to_inception = os.path.join(*[os.path.dirname(os.path.abspath(__file__)),
                                   '..', 'models', 'research', 'inception'])
sys.path.append(path_to_inception)

import tensorflow as tf
import horovod.tensorflow as hvd
from tensorflow.python.client import device_lib
from tensorflow.python.ops import data_flow_ops
from inception.imagenet_data import ImagenetData
from inception import inception_model as inception
from inception.image_processing import batch_inputs

tf.flags.DEFINE_string('checkpoint_dir', 'checkpoint_dir', 'the directory to sore training check poinrts.')
#tf.flags.DEFINE_integer('num_ps', 1, 'number of parameters servers')
tf.flags.DEFINE_string('ps_worker', 'ps', 'number of parameters servers')
tf.flags.DEFINE_integer('num_steps', 1, 'number of forward + back propigation steps')
tf.flags.DEFINE_integer('num_classes', 1000, 'Number of syssets in dataset')
tf.flags.DEFINE_string('server_protocol', 'grpc', 'protocol for servers')
'grpc+mpi'
tf.flags.DEFINE_float('initial_learning_rate', 0.01, 'RMS with exp decay learning rate')
tf.app.flags.DEFINE_float('num_epochs_per_decay', 30.0,
                          """Epochs after which learning rate decays.""")
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.16,
                          """Learning rate decay factor.""")

FLAGS = tf.flags.FLAGS

# Constants dictating the learning rate schedule.
RMSPROP_DECAY = 0.9                # Decay term for RMSProp.
RMSPROP_MOMENTUM = 0.9             # Momentum in RMSProp.
RMSPROP_EPSILON = 1.0              # Epsilon term for RMSProp.

#tf.flags.DEFINE_integer('num_preprocess_threads', 1, 'number of threads')
#tf.flags.DEFINE_integer('batch_size', 32, 'number of threads')
#tf.flags.DEFINE_string('data_dir', None, """Path to dataset in TFRecord format
#                       (aka Example protobufs).""")

DIST_TF_PORT_START = 2222

DIST_TF_WORKER_PORT = 2222
DIST_TF_PS_PORT = 2223

def inception_imagenet_distributed_train():
  '''
  This function is a boiler plate to illistrate the main steps in training a inference
  graph in distributed TensorFlow
  '''
  PRINT_SUMMERY_EVERY = 1
  hvd.init()
  num_workers = hvd.size()
  rank = hvd.rank()
  local_rank = hvd.local_rank()

  #num_workers = 1
  #rank = 0

  print('Running worker %s: total number of wokers %s' % (rank, num_workers))

  with tf.Graph().as_default():

      print('worker %s: Building: Graph' % rank)
      _ps_fn = lambda x: 1
      # greedy = tf.contrib.training.GreedyLoadBalancingStrategy(NUM_PS, _ps_fn)
      greedy = None
      with tf.device('cpu:0'):
          dataset = ImagenetData(subset='train')
          print("worker %s: Building: Step 1" % rank)
          images, labels = batch_inputs(
            dataset, FLAGS.batch_size*num_workers, train=True,
            num_preprocess_threads=FLAGS.num_preprocess_threads,
            num_readers=FLAGS.num_readers)

          print("worker %s: Building: Step 2" % rank)
          images_splits = tf.split(axis=0, num_or_size_splits=num_workers, value=images)
          labels_splits = tf.split(axis=0, num_or_size_splits=num_workers, value=labels)
      print("worker %s: Building: Step 3" % rank)
      global_step = tf.contrib.framework.get_or_create_global_step()

      print("worker %s: Building: Step 4" % rank)
      #optimizer = tf.train.AdagradOptimizer(FLAGS.initial_learning_rate)

      lr = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                      global_step,
                                      2,
                                      FLAGS.learning_rate_decay_factor,
                                      staircase=True)
      lr = FLAGS.initial_learning_rate
      optimizer = tf.train.GradientDescentOptimizer(lr)
      
      #optimizer = tf.train.RMSPropOptimizer(lr, RMSPROP_DECAY,
      #                                momentum=RMSPROP_MOMENTUM,
      #                                epsilon=RMSPROP_EPSILON)
      optimizer = hvd.DistributedOptimizer(optimizer, device_sparse='/cpu:0')

      # When fine-tuning a model, we do not restore the logits but instead we
      # randomly initialize the logits. The number of classes in the output of the
      # logit is the number of classes in specified Dataset.

      # Build inference Graph.
      with tf.name_scope('Inception_Inference') as name_scope:
        print("worker %s: Building: Step 5" % rank)
        try:
          logits = inception.inference(
            images_splits[rank],
            FLAGS.num_classes,
            for_training=True,
            restore_logits=False,
            build_summeries=False
            )
        except TypeError:
          print('accepted TypeError, building model with summeries')
          logits = inception.inference(
            images_splits[rank],
            FLAGS.num_classes,
            for_training=True,
            restore_logits=False,
            )
          
      with tf.name_scope('Inception_Loss') as name_scope:
          split_batch_size = images_splits[rank].get_shape().as_list()[0]
          print('worker %s: split_batch_size %s' % (rank,split_batch_size))
          num_classes = logits[0].get_shape()[-1].value

          onehot_labels = tf.one_hot(tf.cast(labels_splits[rank], tf.int32), depth=num_classes)

          with tf.name_scope('xentropy'):
            print("worker %s: Building: Step 6" % rank)
            cross_entropy = tf.losses.softmax_cross_entropy(
              logits=logits[0], onehot_labels=onehot_labels,
              label_smoothing=.5)
            loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')

          with tf.name_scope('aux_xentropy'):
            print("worker %s: Building: Step 7" % rank)
            aux_cross_entropy = tf.losses.softmax_cross_entropy(
              logits=logits[1], onehot_labels=onehot_labels,
              label_smoothing=.5)
            aux_loss = 0.4 * tf.reduce_mean(aux_cross_entropy, name='aux_loss')

      print("worker %s: Building: Step 8" % rank)
      minimize_opt = optimizer.minimize(loss, global_step=global_step)
      
      hooks = [hvd.BroadcastGlobalVariablesHook(0),
               tf.train.StopAtStepHook(last_step=FLAGS.num_steps)]

      checkpoint_dir = FLAGS.checkpoint_dir if rank == 0 else None

      config = tf.ConfigProto(
          allow_soft_placement=True,
          #log_device_placement=True
          )
      config.gpu_options.visible_device_list = str(hvd.local_rank())
      
      benchmark_delta_list = []
      benchmark_loss_list = []
      step_count = 0
      print('worker %s: Entering MonitoredTrainingSession()' % rank)
      master_target = ''
      config=None
      is_chief = True
      
      with tf.train.MonitoredTrainingSession(checkpoint_dir=FLAGS.checkpoint_dir,
                                             #save_summaries_secs=1800,
                                             save_summaries_steps=PRINT_SUMMERY_EVERY,
                                             config=config,
                                             hooks=hooks) as mon_sess:
          print("worker %s: In MonitoredTrainingSession() context" % rank)
          tf.train.start_queue_runners(sess=mon_sess)
          print("worker %s: Starting training steps" % rank)
          while not mon_sess.should_stop():
              step_count += 1
              last_g_s = 0
              tick = dt.now()
              if step_count % PRINT_SUMMERY_EVERY == 0:
                  _, l, g_s = mon_sess.run((minimize_opt, loss, global_step))
              else:
                  _, l, = mon_sess.run((minimize_opt, loss))
              tock = dt.now()

              benchmark_delta_list.append(float((tock - tick).total_seconds()))
              benchmark_loss_list.append(l)
              if step_count % PRINT_SUMMERY_EVERY == 0:
                  mean_time_delta = np.mean(benchmark_delta_list)
                  mean_loss = np.mean(benchmark_loss_list)
                  global_ex_s = (g_s - last_g_s)*split_batch_size/float(sum(benchmark_delta_list))
                  last_g_s = g_s
                  print("worker %s: l-step %s: g-step %s: l-mean loss %.3f: l-mean example/sec %.3f: g-mean example/sec %.3f:" % (rank, step_count, g_s, mean_loss, split_batch_size/mean_time_delta, global_ex_s))
                  benchmark_delta_list = []
                  benchmark_loss_list = []

      print("worker %s: Finished In MonitoredTrainingSession(): Final loss, %s" % (rank, l))

if __name__ == "__main__":
    inception_imagenet_distributed_train()    
