print('Loading1')
import ml_comm as mc
from functools import reduce
import socket
import os
import sys
import time
import numpy as np
from datetime import datetime as dt
import random
from itertools import groupby
from itertools import chain
print('Loading2')
path_to_inception = os.path.join(*[os.path.dirname(os.path.abspath(__file__)),
                                   '..', 'models', 'research', 'inception'])
sys.path.append(path_to_inception)
print(os.listdir(path_to_inception))
print(os.path.dirname(os.path.abspath(__file__)))
print('Loading3')
import tensorflow as tf
from tensorflow.python.ops import data_flow_ops
from inception.imagenet_data import ImagenetData
from inception import inception_model as inception
from inception.image_processing import batch_inputs
print('Loading4')
tf.flags.DEFINE_string('checkpoint_dir', 'checkpoint_dir', 'the directory to sore training check poinrts.')
tf.flags.DEFINE_string('ps_worker', 'ps', 'Parameter server or worker')
tf.flags.DEFINE_integer('num_ps', 1, 'number of parameters servers')
tf.flags.DEFINE_integer('num_steps', 1, 'number of forward + back propigation steps')
tf.flags.DEFINE_integer('num_classes', 1000, 'Number of syssets in dataset')
tf.flags.DEFINE_string('server_protocol', 'grpc', 'protocol for servers')
tf.flags.DEFINE_string('model', 'inception_v3', '')
'grpc+mpi'
tf.flags.DEFINE_float('initial_learning_rate', 0.4, 'RMS with exp decay learning rate')
tf.app.flags.DEFINE_float('num_epochs_per_decay', 30.0,
                          """Epochs after which learning rate decays.""")
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 2.71,
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
print('Loading5')



def inception_imagenet_distributed_train():
  '''
  This function is a boiler plate to illistrate the main steps in training a inference
  graph in distributed TensorFlow
  '''
  print('Starting inception_imagenet_distributed_train()')
  PRINT_SUMMERY_EVERY = 10
  NUM_STEPS_PER_EPOCH = int(385455/FLAGS.batch_size)

  my_hostname = socket.gethostname()

  num_workers = 8
  rank = 0

  print('Running worker %s: on host %s with mpi rank %d' % (rank, my_hostname, rank))
  with tf.Graph().as_default():

    print('worker %s: Building: Graph' % rank)

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

    global_step = tf.train.get_or_create_global_step() #tf.contrib.framework.get_or_create_global_step()

    print("worker %s: Building: Step 3" % rank)
    #optimizer = tf.train.AdagradOptimizer(0.01)

    lr = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                    global_step,
                                    NUM_STEPS_PER_EPOCH*2,
                                    FLAGS.learning_rate_decay_factor,
                                    staircase=True)
    #optimizer = tf.train.GradientDescentOptimizer(lr)

    optimizer = tf.train.MomentumOptimizer(lr,
                                           momentum=RMSPROP_MOMENTUM,
                                           use_nesterov=True)

    #optimizer = tf.train.RMSPropOptimizer(lr, RMSPROP_DECAY,
    #                                momentum=RMSPROP_MOMENTUM,
    #                                epsilon=RMSPROP_EPSILON)

    # When fine-tuning a model, we do not restore the logits but instead we
    # randomly initialize the logits. The number of classes in the output of the
    # logit is the number of classes in specified Dataset.

    # Build inference Graph.
    with tf.name_scope('Inception_Inference') as name_scope:
      print("worker %s: Building: Step 4" % rank)
      logits = inception.inference(
            images_splits[rank],
            FLAGS.num_classes,
            for_training=True,
            restore_logits=False,
            )
    print("worker %s: Init cray var" % rank)
    tot_model_size = sum([reduce(lambda x, y : x*y, v.get_shape().as_list()) for v in tf.trainable_variables()])
    mc.init(1, 1, tot_model_size, "tensorflow")

    mc.config_team(0,0,100, FLAGS.num_steps, 2, 1)

    FLAGS.checkpoint_dir = FLAGS.checkpoint_dir + '/rank' + str(mc.get_rank())
    rank = mc.get_rank()

    class BcastTensors(tf.train.SessionRunHook):
      def __init__(self):
        self.bcast = None

      def begin(self):
        new_vars = mc.broadcast(tf.trainable_variables(), 0)
        self.bcast = tf.group(*[tf.assign(v, new_vars[k]) for k, v  in enumerate(tf.trainable_variables())])


    with tf.name_scope('Inception_Loss') as name_scope:
      split_batch_size = images_splits[rank].get_shape().as_list()[0]

      num_classes = logits[0].get_shape()[-1].value

      onehot_labels = tf.one_hot(tf.cast(labels_splits[rank], tf.int32), depth=num_classes)

      with tf.name_scope('xentropy'):
        print("worker %s: Building: Step 5" % rank)
        cross_entropy = tf.losses.softmax_cross_entropy(
          logits=logits[0], onehot_labels=onehot_labels,
          label_smoothing=.5, reduction=tf.losses.Reduction.SUM)
        loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')

      with tf.name_scope('aux_xentropy'):
        print("worker %s: Building: Step 6" % rank)
        aux_cross_entropy = tf.losses.softmax_cross_entropy(
          logits=logits[1], onehot_labels=onehot_labels,
          label_smoothing=.5, reduction=tf.losses.Reduction.SUM)
        aux_loss = 0.3 * tf.reduce_mean(aux_cross_entropy, name='aux_loss')

      #regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
      regularization_losses = []
      total_loss = tf.add_n([loss, aux_loss] + regularization_losses, name='total_loss')

    with tf.name_scope('Optimizer'):
      grads_and_vars = optimizer.compute_gradients(total_loss)
      grads = mc.gradients([gv[0] for gv in grads_and_vars], 0)
      gs_and_vs = [(g,v) for (_,v), g in zip(grads_and_vars, grads)]

      train_op = optimizer.apply_gradients(gs_and_vs, global_step=global_step)


    hooks = [tf.train.StopAtStepHook(last_step=FLAGS.num_steps), BcastTensors()]

    config = tf.ConfigProto(
      allow_soft_placement=True,
      #log_device_placement=True
      )
    benchmark_delta_list = []
    benchmark_loss_list = []
    step_count = 0
    print('worker %s: Entering MonitoredTrainingSession()' % rank)
      
    with tf.train.MonitoredTrainingSession(checkpoint_dir=FLAGS.checkpoint_dir,
                                           save_summaries_steps=20,
                                           save_checkpoint_secs=120,
                                           config=config,
                                           hooks=hooks) as mon_sess:
      print("worker %s: In MonitoredTrainingSession() context" % rank)
      tf.train.start_queue_runners(sess=mon_sess)

      print("worker %s: Starting training steps" % rank)
      while not mon_sess.should_stop():
        step_count += 1

        tick = dt.now()
        #if step_count % PRINT_SUMMERY_EVERY == 0:
        if True:
          _, l, g_s = mon_sess.run((train_op, total_loss, global_step))
        else:
          _, l, = mon_sess.run((train_opt, total_loss))
        tock = dt.now()

        benchmark_delta_list.append(float((tock - tick).total_seconds()))
        benchmark_loss_list.append(l)
        #if step_count % PRINT_SUMMERY_EVERY == 0:
        if True:
          mean_time_delta = np.mean(benchmark_delta_list)
          mean_loss = np.mean(benchmark_loss_list)
          print("worker %s: local step %s: global step %s: mean loss %.3f: time %s: mean example/sec %.3f:" % (rank, step_count, g_s, mean_loss, tick, split_batch_size/mean_time_delta))
          benchmark_delta_list = []
          benchmark_loss_list = []

    print("worker %s: Finished In MonitoredTrainingSession(): Final loss, %s" % (rank, l))

  mc.finalize()

if __name__ == "__main__":
    inception_imagenet_distributed_train()    
