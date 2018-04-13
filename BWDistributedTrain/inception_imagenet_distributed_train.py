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
print(os.listdir(path_to_inception))
print(os.path.dirname(os.path.abspath(__file__)))

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

  PRINT_SUMMERY_EVERY = 30

  from  mpi4py import MPI
  comm = MPI.COMM_WORLD
  my_hostname = socket.gethostname()

  rank = comm.Get_rank()
  hostnames = comm.allgather(my_hostname)
  if False:
    NUM_PS = 4
    if rank < NUM_PS:
      job_name = 'ps'
      tf_index = int(rank)
    else:
      job_name = 'worker'
      tf_index = int(rank) - NUM_PS

    num_workers = len(hostnames) - NUM_PS

    tf_ports = list(chain( *[ map(str, range(DIST_TF_PORT_START, DIST_TF_PORT_START+i))
                        for i in [ len(list(group))
                                  for _, group in groupby(hostnames) ]] ))

    tf_ps_hosts_ports = [':'.join(hn_p)
                         for hn_p in zip(
        hostnames[:NUM_PS], tf_ports[:NUM_PS]
        )]

    # e.g tf_worker_hosts_ports = 'nid25428:2222'

    tf_worker_hosts_ports = [':'.join(hn_p)
                             for hn_p in zip(
        hostnames[NUM_PS:], tf_ports[NUM_PS:]
        )]

    # e.g tf_worker_hosts_ports = 'nid25429:2222,nid25430:2222'
  else:
    host_rank_dict = {k: [i[0] for i in g]
                      for k, g in groupby(sorted(enumerate(hostnames), key = lambda y: y[1]),
                                          key = lambda x: x[1])}
    host_rank_dict_values = list(host_rank_dict.values())
    print("host_rank_dict_values %s" % str(host_rank_dict_values))
    if len(host_rank_dict_values) < 2:
      ps_ranks, worker_ranks = host_rank_dict_values[0][:2]
      ps_ranks = [ps_ranks]
      worker_ranks = [worker_ranks]
    else:
      ps_ranks, worker_ranks =  list(zip(*host_rank_dict_values))[:2]
    unique_hostnames = list(host_rank_dict.keys())

    NUM_PS = len(unique_hostnames)

    tf_ps_hosts_ports = [':'.join(hn_p)
                         for hn_p in zip(
        unique_hostnames[:NUM_PS], [str(DIST_TF_PS_PORT)]*NUM_PS
        )]

    tf_worker_hosts_ports = [':'.join(hn_p)
                             for hn_p in zip(
        unique_hostnames, [str(DIST_TF_WORKER_PORT)]*len(unique_hostnames)
                               )]
    do_nothing = False
    if rank in ps_ranks:
      job_name = 'ps'
      tf_index = ps_ranks.index(rank)
      if tf_index >= NUM_PS:
        do_nothing = True
      os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


    elif rank in worker_ranks:
      job_name = 'worker'
      tf_index = worker_ranks.index(rank)
    else:
      do_nothing = True

    if do_nothing:
      print('My rank is %(r)s on host %(hn)s and I have nothing to do.\n'
            '(%(r)s, %(hn)s) Waiting for my fellow processing elements '
            'to finish then exiting gracefully' % {'r': rank, 'hn': my_hostname})
      comm.barrier()
      comm.barrier()

    num_workers = len(unique_hostnames)
    
  import tensorflow as tf
  from tensorflow.python.ops import data_flow_ops
  from inception.imagenet_data import ImagenetData
  from inception import inception_model as inception
  from inception.image_processing import batch_inputs
  
  tf.flags.DEFINE_string('checkpoint_dir', 'checkpoint_dir', 'the directory to sore training check poinrts.')
  tf.flags.DEFINE_integer('num_ps', 1, 'number of parameters servers')
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
        dataset, FLAGS.batch_size, train=True,
        num_preprocess_threads=FLAGS.num_preprocess_threads,
        num_readers=FLAGS.num_readers)
      print("worker %s: Building: Step 2" % tf_index)
      images_splits = tf.split(axis=0, num_or_size_splits=num_workers, value=images)
      labels_splits = tf.split(axis=0, num_or_size_splits=num_workers, value=labels)
      print("worker %s: Building: Step 3" % tf_index)
      global_step = tf.contrib.framework.get_or_create_global_step()

      print("worker %s: Building: Step 3" % tf_index)
      #optimizer = tf.train.AdagradOptimizer(0.01)

      lr = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                      global_step,
                                      2,
                                      FLAGS.learning_rate_decay_factor,
                                      staircase=True)
      #optimizer = tf.train.GradientDescentOptimizer(lr)
      
      optimizer = tf.train.RMSPropOptimizer(lr, RMSPROP_DECAY,
                                      momentum=RMSPROP_MOMENTUM,
                                      epsilon=RMSPROP_EPSILON)

      # When fine-tuning a model, we do not restore the logits but instead we
      # randomly initialize the logits. The number of classes in the output of the
      # logit is the number of classes in specified Dataset.

      # Build inference Graph.
      with tf.name_scope('Inception_Inference') as name_scope:
        print("worker %s: Building: Step 4" % tf_index)
        logits = inception.inference(
              images_splits[tf_index],
              FLAGS.num_classes,
              for_training=True,
              restore_logits=False,
              )
      with tf.name_scope('Inception_Loss') as name_scope:
          split_batch_size = images_splits[tf_index].get_shape().as_list()[0]

          num_classes = logits[0].get_shape()[-1].value

          onehot_labels = tf.one_hot(tf.cast(labels_splits[tf_index], tf.int32), depth=num_classes)

          with tf.name_scope('xentropy'):
            print("worker %s: Building: Step 5" % tf_index)
            cross_entropy = tf.losses.softmax_cross_entropy(
              logits=logits[0], onehot_labels=onehot_labels,
              label_smoothing=.5)
            loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')

          with tf.name_scope('aux_xentropy'):
            print("worker %s: Building: Step 6" % tf_index)
            aux_cross_entropy = tf.losses.softmax_cross_entropy(
              logits=logits[1], onehot_labels=onehot_labels,
              label_smoothing=.5)
            aux_loss = 0.4 * tf.reduce_mean(aux_cross_entropy, name='aux_loss')

      print("worker %s: Building: Step 7" % tf_index)
      minimize_opt = optimizer.minimize(loss, global_step=global_step)

    hooks = [tf.train.StopAtStepHook(last_step=FLAGS.num_steps)]

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
          print("worker %s: local step %s: global step %s: mean loss %.3f: time %s: mean example/sec %.3f:" % (tf_index, step_count, g_s, mean_loss, tick, split_batch_size/mean_time_delta))
          benchmark_delta_list = []
          benchmark_loss_list = []

    print("worker %s: Finished In MonitoredTrainingSession(): Final loss, %s" % (tf_index, l))

  comm.barrier()


if __name__ == "__main__":
    inception_imagenet_distributed_train()    
