import socket
import sys
import os
import re
from itertools import groupby

import tensorflow as tf
from  mpi4py import MPI

path_to_inception = os.path.join(*[os.path.dirname(os.path.abspath(__file__)),
                                   '..', 'models', 'research', 'inception'])
sys.path.append(path_to_inception)

from inception.imagenet_data import ImagenetData
from inception import inception_model as inception
from inception.image_processing import batch_inputs

tf.flags.DEFINE_string('checkpoint_dir', 'checkpoint_dir', 'the directory to sore training check poinrts.')
tf.flags.DEFINE_string('ps_worker', 'ps', 'Parameter server or worker')
tf.flags.DEFINE_string('server_protocol', 'grpc', 'protocol for servers')

FLAGS = tf.flags.FLAGS

DIST_TF_PORT = 2222

def restore_graph():

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
    dataset = ImagenetData(subset='train')
    images, labels = batch_inputs(
        dataset, FLAGS.batch_size*num_workers, train=True,
        num_preprocess_threads=FLAGS.num_preprocess_threads,
        num_readers=FLAGS.num_readers)

    print("worker %s: Building: Step 2" % tf_index)

    images_splits = tf.split(axis=0, num_or_size_splits=num_workers, value=images)
    labels_splits = tf.split(axis=0, num_or_size_splits=num_workers, value=labels)    

    with tf.name_scope('Inception_Inference') as name_scope:
        print("worker %s: Building: Step 4" % tf_index)
        logits = inception.inference(
            images_splits[tf_index],
            1000,
            for_training=True,
            restore_logits=False,
            )

    print('DIR {}'.format(os.listdir(FLAGS.checkpoint_dir)))
    with tf.train.MonitoredTrainingSession(master=server.target,
                                           is_chief=(tf_index == 0)) as sess:
        last_model = sorted([ l for l in os.listdir(FLAGS.checkpoint_dir) if re.match(r'model.*\.meta', l) is not None ])[1]
        #saver = tf.train.import_meta_graph(os.path.join(FLAGS.checkpoint_dir, last_model))
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint('./'))

if __name__ == "__main__":
    restore_graph()
