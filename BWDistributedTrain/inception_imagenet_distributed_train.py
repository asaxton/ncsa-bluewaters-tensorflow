print('Loading1')
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
tf.app.flags.DEFINE_boolean('restore', False, 'restores a checkpoint and saves a usable saved graph')
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
DIST_TF_PORT_START = 2222

DIST_TF_WORKER_PORT = 2222
DIST_TF_PS_PORT = 2223
DIST_TF_PORT = 2222
def inception_imagenet_distributed_train():
  '''
  This function is a boiler plate to illistrate the main steps in training a inference
  graph in distributed TensorFlow
  '''
  print('Starting inception_imagenet_distributed_train()')
  PRINT_SUMMERY_EVERY = 10
  NUM_STEPS_PER_EPOCH = int(385455/FLAGS.batch_size)
  from  mpi4py import MPI
  comm = MPI.COMM_WORLD
  my_hostname = socket.gethostname()

  rank = comm.Get_rank()

  ps_worker_hostname_list = comm.allgather((FLAGS.ps_worker, my_hostname))

  rank_ps_worker_hostname_list = sorted(enumerate(ps_worker_hostname_list), key=lambda x: x[1][0])
  ranks, ps_worker_hostname_list = list(zip(*rank_ps_worker_hostname_list))
  ps_worker, hostname_list = list(zip(*ps_worker_hostname_list))
  rank_ps_worker_hostname_list = list(zip(ranks, ps_worker, hostname_list))

  ps_rank_hostname, worker_rank_hostname = [list(g) for k, g in groupby(rank_ps_worker_hostname_list, lambda x: x[1])]
  ps_rank_hostname = list(zip(*ps_rank_hostname))
  worker_rank_hostname = list(zip(*worker_rank_hostname))


  ps_ranks = ps_rank_hostname[0]
  worker_ranks = worker_rank_hostname[0]

  ps_hostnames = ps_rank_hostname[2]
  worker_hostnames = worker_rank_hostname[2]

  comm.barrier()
  print('ps_ranks %s' % str(ps_ranks))
  print('worker_ranks %s' % str(worker_ranks))
  comm.barrier()

  num_workers = len(worker_ranks)
  num_ps = len(ps_ranks)

  tf_ps_hosts_ports = [':'.join(hn_p)
                       for hn_p in zip(
          ps_hostnames, [str(DIST_TF_PORT)]*num_ps
          )]

  tf_worker_hosts_ports = [':'.join(hn_p)
                           for hn_p in zip(
          worker_hostnames, [str(DIST_TF_PORT)]*num_workers
      )]

  # e.g tf_worker_hosts_ports = 'nid25428:2222'

  do_nothing = False

  if rank in ps_ranks:
    job_name = 'ps'
    tf_index = ps_ranks.index(rank)
    #os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
  elif rank in worker_ranks:
    job_name = 'worker'
    tf_index = worker_ranks.index(rank)
  else:
      do_nothing = True

  comm.barrier()
  print("tf_ps_hosts_ports, %s " % tf_ps_hosts_ports)
  print("tf_worker_hosts_ports, %s" % tf_worker_hosts_ports)
  print("my_hostname %s, tf_index %s, job_name %s " % (my_hostname, tf_index, job_name))
  comm.barrier()

  if do_nothing:
      print('doing nothing')
      comm.barrier()
      comm.barrier()
      comm.barrier()
      return




  print("tf_ps_hosts_ports, %s" % tf_ps_hosts_ports)
  print("tf_worker_hosts_ports, %s" % tf_worker_hosts_ports)
  
  cluster = tf.train.ClusterSpec({'ps': tf_ps_hosts_ports,
                                  'worker': tf_worker_hosts_ports})


  server = tf.train.Server(cluster, job_name=job_name,
                           task_index=tf_index,
                           protocol=FLAGS.server_protocol)

  if job_name == 'ps':
    print('Running parameter server %s: on host %s with mpi rank %d' % (tf_index, my_hostname, rank))
    comm.barrier() # server.join()
    comm.barrier()
    return


  print('Running worker %s: on host %s with mpi rank %d' % (tf_index, my_hostname, rank))
  with tf.Graph().as_default():

    print('worker %s: Building: Graph' % tf_index)
    with tf.device(tf.train.replica_device_setter( cluster=cluster, worker_device='/job:worker/task:%s' % tf_index )):
      dataset = ImagenetData(subset='train')
      print("worker %s: Building: Step 1" % tf_index)
      images, labels = batch_inputs(
        dataset, FLAGS.batch_size*num_workers, train=True,
        num_preprocess_threads=FLAGS.num_preprocess_threads,
        num_readers=FLAGS.num_readers)
      print("worker %s: Building: Step 2" % tf_index)
      images_splits = tf.split(axis=0, num_or_size_splits=num_workers, value=images)
      labels_splits = tf.split(axis=0, num_or_size_splits=num_workers, value=labels)
      print("worker %s: Building: Step 3" % tf_index)

      global_step = tf.train.get_or_create_global_step() #tf.contrib.framework.get_or_create_global_step()

      print("worker %s: Building: Step 3" % tf_index)
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
      with tf.variable_scope('Inception_Inference') as variable_scope:
        print("worker %s: Building: Step 4" % tf_index)
        logits = inception.inference(
              images_splits[tf_index],
              FLAGS.num_classes,
              for_training=True,
              restore_logits=False,
              )
      with tf.variable_scope('Inception_Loss') as variable_scope:
          split_batch_size = images_splits[tf_index].get_shape().as_list()[0]

          num_classes = logits[0].get_shape()[-1].value

          onehot_labels = tf.one_hot(tf.cast(labels_splits[tf_index], tf.int32), depth=num_classes)

          with tf.variable_scope('xentropy'):
            print("worker %s: Building: Step 5" % tf_index)
            cross_entropy = tf.losses.softmax_cross_entropy(
              logits=logits[0], onehot_labels=onehot_labels,
              label_smoothing=.5) #, reduction=tf.losses.Reduction.SUM)
            loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')

          with tf.variable_scope('aux_xentropy'):
            print("worker %s: Building: Step 6" % tf_index)
            aux_cross_entropy = tf.losses.softmax_cross_entropy(
              logits=logits[1], onehot_labels=onehot_labels,
              label_smoothing=.5, reduction=tf.losses.Reduction.SUM)
            aux_loss = 0.3 * tf.reduce_mean(aux_cross_entropy, name='aux_loss')

          #regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
          regularization_losses = []
          total_loss = tf.add_n([loss, aux_loss] + regularization_losses, name='total_loss')

      print("worker %s: Building: Step 7" % tf_index)
      with tf.variable_scope('Optimizer'):
        grads_and_vars = optimizer.compute_gradients(total_loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    #minimize_opt = optimizer.minimize(total_loss, global_step=global_step)
    if FLAGS.restore:
      print('Restoring checkpoint, saving a final model, then exiting')
      hooks = [SaveAtEnd(), tf.train.StopAtStepHook(last_step=0)]
    else:
      hooks = [SaveAtEnd(), tf.train.StopAtStepHook(last_step=FLAGS.num_steps)]

    config = tf.ConfigProto(
      allow_soft_placement=True,
      #log_device_placement=True
      )
    benchmark_delta_list = []
    benchmark_loss_list = []
    step_count = 0

    print('worker %s: Entering MonitoredTrainingSession()' % tf_index)

    with tf.train.MonitoredTrainingSession(master=server.target,
                                           is_chief=(tf_index == 0),
                                           checkpoint_dir=FLAGS.checkpoint_dir,
                                           #save_summaries_secs=1800,
                                           save_summaries_steps=PRINT_SUMMERY_EVERY,
                                           config=config,
                                           hooks=hooks) as mon_sess:
      print("worker %s: In MonitoredTrainingSession() context" % tf_index)
      tf.train.start_queue_runners(sess=mon_sess)
      comm.barrier()
      print("worker %s: Starting training steps" % tf_index)
      while not mon_sess.should_stop():
        step_count += 1

        tick = dt.now()
        #if step_count % PRINT_SUMMERY_EVERY == 0:
        if True:
          _, l, g_s = mon_sess.run((train_op, total_loss, global_step))
        else:
          _, l, = mon_sess.run((train_op, total_loss))
        tock = dt.now()

        benchmark_delta_list.append(float((tock - tick).total_seconds()))
        benchmark_loss_list.append(l)
        #if step_count % PRINT_SUMMERY_EVERY == 0:
        if True:
          mean_time_delta = np.mean(benchmark_delta_list)
          mean_loss = np.mean(benchmark_loss_list)
          print("worker %s: local step %s: global step %s: mean loss %.3f: time %s: mean example/sec %.3f:" % (tf_index, step_count, g_s, mean_loss, tick, split_batch_size/mean_time_delta))
          benchmark_delta_list = []
          benchmark_loss_list = []

    print("worker %s: Finished In MonitoredTrainingSession(): Final loss, %s" % (tf_index, l))
  comm.barrier()

class SaveAtEnd(tf.train.SessionRunHook):
  def begin(self):
    inception_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                       scope='Inception_Inference')
    print('Saving Model Variables. Number of Vars: {}'.format(len(inception_vars)))
    self._saver = tf.train.Saver(inception_vars, save_relative_paths=True)
  def end(self, session):

    self._saver.save(session, os.path.join(FLAGS.checkpoint_dir, 'final_save'))

if __name__ == "__main__":
    inception_imagenet_distributed_train()    
