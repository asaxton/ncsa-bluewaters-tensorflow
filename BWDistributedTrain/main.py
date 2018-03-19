import tensorflow as tf
import socket
import os
import sys
import time
import random

path_to_inception = os.path.join(*[os.path.dirname(os.path.abspath(__file__)),
                                   '..', 'models', 'research', 'inception'])
sys.path.append(path_to_inception)
from itertools import groupby
from itertools import chain
from inception.imagenet_data import ImagenetData

from inception_train import _tower_loss
#from inception.inception_train import _tower_loss
from inception import inception_model as inception

from inception.inception_train import _average_gradients
from inception.image_processing import batch_inputs
from inception.slim import slim

from tensorflow.python.layers.core import Dense

tf.flags.DEFINE_string('checkpoint_dir', 'checkpoint_dir', 'the directory to sore training check poinrts.')
tf.flags.DEFINE_integer('num_ps', 1, 'number of parameters servers')
tf.flags.DEFINE_integer('num_steps', 1, 'number of forward + back propigation steps')
tf.flags.DEFINE_integer('num_classes', 1000, 'Number of syssets in dataset')
tf.flags.DEFINE_string('server_protocol', 'grpc', 'protocol for servers')
#tf.flags.DEFINE_float('initial_learning_rate', 0.1, 'RMS with exp decay learning rate')
# Constants dictating the learning rate schedule.
RMSPROP_DECAY = 0.9                # Decay term for RMSProp.
RMSPROP_MOMENTUM = 0.9             # Momentum in RMSProp.
RMSPROP_EPSILON = 1.0              # Epsilon term for RMSProp.

#tf.flags.DEFINE_integer('num_preprocess_threads', 1, 'number of threads')
#tf.flags.DEFINE_integer('batch_size', 32, 'number of threads')
#tf.flags.DEFINE_string('data_dir', None, """Path to dataset in TFRecord format
#                       (aka Example protobufs).""")

FLAGS = tf.flags.FLAGS

DIST_TF_PORT_START = 2222

DIST_TF_WORKER_PORT = 2222
DIST_TF_PS_PORT = 2223

def super_simple_test():
  '''
  This function is meant to be run in a single thread as a sanity check.
  '''
  TRUE_M = .4
  TRUE_B = -3
  PRINT_LOSS_EVERY_X_STEP = 100
  NUM_TRAIN = 1000
  NUM_VAL = 100
  INPUT_DOMAIN = [0,10]
  NOISE_WIDTH = .1
  NUM_STEPS = 1000
  
  x_data = [[random.uniform(*INPUT_DOMAIN)] for i in xrange(NUM_TRAIN + NUM_VAL)]
  # y_noise = [random.uniform(-NOISE_WIDTH,NOISE_WIDTH) for i in xrange(NUM_TRAIN + NUM_VAL)]
  y_noise = [random.normalvariate(0,NOISE_WIDTH) for i in xrange(NUM_TRAIN + NUM_VAL)]
  y_data = map(lambda w : [w[0][0]*TRUE_M + TRUE_B + w[1]], zip(x_data, y_noise))

  x_train_data = x_data[:NUM_TRAIN]
  y_train_data = y_data[:NUM_TRAIN]

  x_val_data = x_data[NUM_TRAIN:]
  y_val_data = y_data[NUM_TRAIN:]

  with tf.name_scope('Global_Step'):
    global_step = tf.contrib.framework.get_or_create_global_step()
    
  with tf.name_scope('Build_Constant'):
    tf_x_train_data = tf.constant(x_train_data, dtype=tf.float32, name="x_data")
    tf_y_train_data = tf.constant(y_train_data, dtype=tf.float32, name="y_data")

  with tf.name_scope('Dense_Model'):
    linear_model = Dense(units=1)

  with tf.name_scope('Linear_Model'):
    y_pred = linear_model(tf_x_train_data)

  with tf.name_scope('Training_Loss'):
    loss = tf.losses.mean_squared_error(labels=tf_y_train_data,
                                        predictions=y_pred)

  with tf.name_scope('Optimizer'):
    #optimizer = tf.train.AdagradOptimizer(0.1)
    optimizer = tf.train.GradientDescentOptimizer(0.01)

  with tf.name_scope('Global_Initializer'):
    init = tf.global_variables_initializer()

  with tf.name_scope('Minimize'):
    train = optimizer.minimize(loss, global_step=global_step)

  with tf.Session() as sess:
    sess.run(init)
    for i in xrange(NUM_STEPS):
      _, l = sess.run((train,loss))
      if i % PRINT_LOSS_EVERY_X_STEP == 0:
        print "Loss %s" % l
    print "True kernel %s, calculated kernel %s" % (TRUE_M, sess.run(linear_model.kernel))
    print "True bias %s, calculated bias %s" % (TRUE_B, sess.run(linear_model.bias))

  writer = tf.summary.FileWriter('.')
  writer.add_graph(tf.get_default_graph())

def test(server, cluster, tf_index, job_name, num_workers):
  '''
  This function is meant to be run distributed for a sanity check.
  '''
  from  mpi4py import MPI
  comm = MPI.COMM_WORLD

  TRUE_M = .4
  TRUE_B = -3
  PRINT_LOSS_EVERY_X_STEP = 100
  NUM_TRAIN = 1000
  NUM_VAL = 100
  INPUT_DOMAIN = [0,10]
  NOISE_WIDTH = .1

  if job_name == 'ps':
    print 'Running parameter server %s' % tf_index
    print 'tf_index, %s job_name, %s hostname, %s' % (tf_index, job_name, socket.gethostname())
    comm.barrier()
    comm.barrier()
    # server.join() # TF known bug that server.join() 
                    # will only block and not rejoin
                    # when workers are finished.
                    # Using mpi4py to synchronize.
    return
  else:
    print 'tf_index, %s job_name, %s hostname, %s' % (tf_index, job_name, socket.gethostname())

  x_data = [[random.uniform(*INPUT_DOMAIN)] for i in xrange(NUM_TRAIN + NUM_VAL)]
  # y_noise = [random.uniform(-NOISE_WIDTH,NOISE_WIDTH) for i in xrange(NUM_TRAIN + NUM_VAL)]
  y_noise = [random.normalvariate(0,NOISE_WIDTH) for i in xrange(NUM_TRAIN + NUM_VAL)]
  y_data = map(lambda w : [w[0][0]*TRUE_M + TRUE_B + w[1]], zip(x_data, y_noise))

  x_train_data = x_data[:NUM_TRAIN]
  y_train_data = y_data[:NUM_TRAIN]

  x_val_data = x_data[NUM_TRAIN:]
  y_val_data = y_data[NUM_TRAIN:]

  print "x_train_data %s, y_train_data %s" % (len(x_train_data[0]), len(y_train_data[0]))
  print 'Running worker %s' % tf_index
  with tf.device(tf.train.replica_device_setter( cluster=cluster, worker_device='/job:worker/task:%s' % tf_index)):
    with tf.name_scope('Global_Step'):
      global_step = tf.contrib.framework.get_or_create_global_step()

    with tf.name_scope('Build_Constant'):
      tf_x_train_data = tf.constant(x_train_data, dtype=tf.float32, name="x_data")
      tf_y_train_data = tf.constant(y_train_data, dtype=tf.float32, name="y_data")

    with tf.name_scope('Split_Constant'):
      x_data_local = tf.split(
        axis=0,
        num_or_size_splits=num_workers,
        value=tf_x_train_data)[tf_index]

      y_data_local = tf.split(
        axis=0,
        num_or_size_splits=num_workers,
        value=tf_y_train_data)[tf_index]

    with tf.name_scope('Dense_Model'):
      linear_model = Dense(units=1)

    with tf.name_scope('Linear_Model'):
      y_pred = linear_model(x_data_local)

    with tf.name_scope('Training_Loss'):
      loss = tf.losses.mean_squared_error(labels=y_data_local,
                                          predictions=y_pred)

    with tf.name_scope('Optimizer'):
      #optimizer = tf.train.AdagradOptimizer(0.1)
      optimizer = tf.train.GradientDescentOptimizer(0.001)

    with tf.name_scope('Minimize'):
      train = optimizer.minimize(loss, global_step=global_step)

    final_step_ops = []
    with tf.name_scope('Train_Print_Ops'):
      _ = tf.Print(linear_model.kernel,
                   [linear_model.kernel],
                   message="True Kernel %s, Kernel on worker %s " % (TRUE_M, tf_index))
      final_step_ops.append(_)
      _ = tf.Print(linear_model.bias,
                   [linear_model.bias],
                   message="True Bias %s, Bias on worker %s " % (TRUE_B, tf_index))
      final_step_ops.append(_)

    # validation ops
    if tf_index == 0:
      with tf.name_scope('Validation_Ops'):
        tf_val_data_x = tf.constant(x_val_data, dtype=tf.float32, name="x_data")
        tf_val_data_y = tf.constant(y_val_data, dtype=tf.float32, name="y_data")

        y_val_pred = linear_model(tf_val_data_x)

        val_loss = tf.losses.mean_squared_error(labels=tf_val_data_y,
                                                predictions=y_val_pred)

        val_print = tf.Print(val_loss,
                             [val_loss],
                             message='Validation Loss ')
      final_step_ops.extend([val_loss, val_print])

    hooks = [tf.train.StopAtStepHook(last_step=FLAGS.num_steps),
             tf.train.FinalOpsHook(final_step_ops)]
    config = tf.ConfigProto(
      allow_soft_placement=True,
      )

    print 'Run on worker %s' % tf_index
    with tf.train.MonitoredTrainingSession(master=server.target,
                                           is_chief=(tf_index == 0),
                                           checkpoint_dir=FLAGS.checkpoint_dir,
                                           save_summaries_steps=1,
                                           config=config,
                                           hooks=hooks) as mon_sess:
      print "In Monitored training session context"
      step_num = 0
      comm.barrier()
      while not mon_sess.should_stop():
        _, loss_ret = mon_sess.run((train,loss))
        if step_num % PRINT_LOSS_EVERY_X_STEP == 0:
          print "loss in worker %s at step %s: %s" % (tf_index, step_num, str(loss_ret))
        step_num += 1

      print "last loss in worker %s: %s" % (tf_index,str(loss_ret))
    comm.barrier()
    print "Done with ManagedTrainingSession in worker %s" % tf_index

def main(_):
  from  mpi4py import MPI
  comm = MPI.COMM_WORLD
  my_hostname = socket.gethostname()

  node_index = comm.Get_rank()
  hostnames = comm.allgather(my_hostname)
  if False:
    if node_index < FLAGS.num_ps:
      job_name = 'ps'
      tf_index = int(node_index)
    else:
      job_name = 'worker'
      tf_index = int(node_index) - FLAGS.num_ps

    num_workers = len(hostnames) - FLAGS.num_ps

    tf_ports = list(chain( *[ map(str, range(DIST_TF_PORT_START, DIST_TF_PORT_START+i))
                        for i in [ len(list(group))
                                  for _, group in groupby(hostnames) ]] ))

    tf_ps_hosts_ports = [':'.join(hn_p)
                         for hn_p in zip(
        hostnames[:FLAGS.num_ps], tf_ports[:FLAGS.num_ps]
        )]

    # e.g tf_worker_hosts_ports = 'nid25428:2222'

    tf_worker_hosts_ports = [':'.join(hn_p)
                             for hn_p in zip(
        hostnames[FLAGS.num_ps:], tf_ports[FLAGS.num_ps:]
        )]

    # e.g tf_worker_hosts_ports = 'nid25429:2222,nid25430:2222'
  else:
    host_rank_dict = {k: [i[0] for i in g]
                      for k, g in groupby(sorted(enumerate(hostnames), key = lambda y: y[1]),
                                          key = lambda x: x[1])}

    ps_ranks, worker_ranks =  zip(*host_rank_dict.values())[:2]
    unique_hostnames = host_rank_dict.keys()

    tf_ps_hosts_ports = [':'.join(hn_p)
                         for hn_p in zip(
        unique_hostnames[:FLAGS.num_ps], [str(DIST_TF_PS_PORT)]*FLAGS.num_ps
        )]

    # e.g tf_worker_hosts_ports = 'nid25428:2222'

    tf_worker_hosts_ports = [':'.join(hn_p)
                             for hn_p in zip(
        unique_hostnames, [str(DIST_TF_WORKER_PORT)]*len(unique_hostnames)
                               )]
    do_nothing = False
    if node_index in ps_ranks:
      job_name = 'ps'
      tf_index = ps_ranks.index(node_index)
      if tf_index >= FLAGS.num_ps:
        do_nothing = True
    elif node_index in worker_ranks:
      job_name = 'worker'
      tf_index = worker_ranks.index(node_index)
    else:
      do_nothing = True

    if do_nothing:
      print('My rank is %s on host %s. I have nothing to do.\n'
            'Waiting for my bretherin to finish then exiting gracefully' % (node_index, my_hostname))
      comm.barrier()
      comm.barrier()

    num_workers = len(unique_hostnames)
    
  print "tf_ps_hosts_ports, ", tf_ps_hosts_ports
  print "tf_worker_hosts_ports, ", tf_worker_hosts_ports

  print "tf_ps_hosts_ports, ", tf_ps_hosts_ports
  print "tf_worker_hosts_ports, ", tf_worker_hosts_ports

  cluster = tf.train.ClusterSpec({'ps': tf_ps_hosts_ports,
                                  'worker': tf_worker_hosts_ports})

  server = tf.train.Server(cluster, job_name=job_name,
                           task_index=tf_index,
                           protocol=FLAGS.server_protocol)


  test(server, cluster, tf_index, job_name, num_workers)

if __name__ == '__main__':
  tf.app.run()
