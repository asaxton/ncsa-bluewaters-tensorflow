import random
import socket
import tensorflow as tf
from itertools import groupby
from itertools import islice

from tensorflow.python.layers.core import Dense

from  mpi4py import MPI

comm = MPI.COMM_WORLD
my_hostname = socket.gethostname()

rank = comm.Get_rank()
hostnames = comm.allgather(my_hostname)

DIST_TF_PORT = 2222

def simple_regression():
    NUM_PS = 1
    PRINT_LOSS_EVERY_X_STEP = 20

    TRUE_M = 0.3
    TRUE_B = -3.
    NUM_STEPS=1000
    INPUT_DOMAIN = [0,10]
    NUM_TRAIN = 1000
    NUM_VAL = 100
    NOISE_WIDTH = .1

    host_rank_dict = {k: [i[0] for i in g]
                      for k, g in groupby(sorted(enumerate(hostnames), key = lambda y: y[1]),
                                          key = lambda x: x[1])}

    unique_hostnames, _  = list(zip(*sorted(host_rank_dict.items(), key=lambda x: x[0])))
    usable_ranks = list(zip(*_))[0]
    print('host_rank_dict %s' % host_rank_dict)
    print('usable ranks %s' % str(usable_ranks))
    ps_ranks = usable_ranks[:NUM_PS]
    worker_ranks = usable_ranks[NUM_PS:]

    ps_hostnames = unique_hostnames[:NUM_PS]
    worker_hostnames = unique_hostnames[NUM_PS:]

    comm.barrier()
    print('ps_ranks %s' % str(ps_ranks))
    print('worker_ranks %s' % str(worker_ranks))
    comm.barrier()

    num_workers = len(worker_ranks)

    tf_ps_hosts_ports = [':'.join(hn_p)
                         for hn_p in zip(
            ps_hostnames, [str(DIST_TF_PORT)]*NUM_PS
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

    cluster = tf.train.ClusterSpec({'ps': tf_ps_hosts_ports,
                                    'worker': tf_worker_hosts_ports})

    server = tf.train.Server(cluster, job_name=job_name,
                             task_index=tf_index,
                             protocol='grpc')

    if job_name == 'ps':
        print('Running parameter server %s' % tf_index)
        print('tf_index, %s job_name, %s hostname, %s' % (tf_index, job_name, socket.gethostname()))
        comm.barrier()
        comm.barrier()
        comm.barrier()
        return
        #server.join() # TF known bug that server.join() 
        # will only block and not rejoin
        # when workers are finished.
        # Using mpi4py to synchronize.

    x_data = [[random.uniform(*INPUT_DOMAIN)] for i in range(NUM_TRAIN + NUM_VAL)]
    y_noise = [random.normalvariate(0,NOISE_WIDTH) for i in range(NUM_TRAIN + NUM_VAL)]
    y_data = list(map(lambda w : [w[0][0]*TRUE_M + TRUE_B + w[1]], zip(x_data, y_noise)))

    x_train_data = x_data[:NUM_TRAIN]
    y_train_data = y_data[:NUM_TRAIN]

    x_val_data = x_data[NUM_TRAIN:]
    y_val_data = y_data[NUM_TRAIN:]

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
      optimizer = tf.train.GradientDescentOptimizer(0.01)

    with tf.name_scope('Minimize'):
      train = optimizer.minimize(loss, global_step=global_step)

    final_step_ops = []
    with tf.name_scope('Train_Print_Ops'):
      _ = tf.Print(linear_model.kernel,
                   [linear_model.kernel],
                   message="\n\nTrue Kernel %s, Kernel on worker %s" % (TRUE_M, tf_index))
      final_step_ops.append(_)
      _ = tf.Print(linear_model.bias,
                   [linear_model.bias],
                   message="\n\nTrue Bias %s, Bias on worker %s" % (TRUE_B, tf_index))
      final_step_ops.append(_)

    hooks = [tf.train.StopAtStepHook(last_step=NUM_STEPS),
             tf.train.FinalOpsHook(final_step_ops)]

    config = tf.ConfigProto(
      allow_soft_placement=True,
      )

    print('Run on %s %s' % (job_name, tf_index))
    comm.barrier()
    with tf.train.MonitoredTrainingSession(master=server.target,
                                           is_chief=(tf_index == 0),
                                           checkpoint_dir='./checkpoint_dir',
                                           save_summaries_steps=1,
                                           config=config,
                                           hooks=hooks) as mon_sess:
        step_num = 0
        comm.barrier()
        while not mon_sess.should_stop():
            _, loss_ret = mon_sess.run((train,loss))
            if step_num % PRINT_LOSS_EVERY_X_STEP == 0:
                print("loss in worker %s at step %s: %s" % (tf_index, step_num, str(loss_ret)))
            step_num += 1
    comm.barrier()
if __name__ == '__main__':
    simple_regression()
