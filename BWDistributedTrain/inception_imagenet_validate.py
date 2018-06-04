import os
import sys
path_to_inception = os.path.join(*[os.path.dirname(os.path.abspath(__file__)),
                                   '..', 'models', 'research', 'inception'])
sys.path.append(path_to_inception)

import tensorflow as tf
from tensorflow.python.ops import data_flow_ops
from inception.imagenet_data import ImagenetData
from inception import inception_model as inception
from inception.image_processing import batch_inputs

tf.flags.DEFINE_string('checkpoint_dir', 'checkpoint_dir', 'the directory to sore training check poinrts.')
tf.flags.DEFINE_integer('num_classes', 1000, 'Number of syssets in dataset')
FLAGS = tf.flags.FLAGS


def inception_imagenet_validate():
    print('tf.reset_default_graph()')
    tf.reset_default_graph()
    print('Stating inception_imagenet_validate()')
    dataset = ImagenetData(subset='validation')    
    print('batch_inputs()')
    images, labels = batch_inputs(
        dataset, FLAGS.batch_size, train=False,
        num_preprocess_threads=FLAGS.num_preprocess_threads,
        num_readers=FLAGS.num_readers)
    print('inception.inference()')
    with tf.variable_scope('Inception_Inference') as variable_scope:    
        logits = inception.inference(
            images,
            FLAGS.num_classes,
            for_training=False,
            restore_logits=True,
            )
    
    num_classes = logits[0].get_shape()[-1].value
    
    onehot_labels = tf.one_hot(tf.cast(labels, tf.int32), depth=num_classes)
    
    with tf.name_scope('xentropy'):
        print("tf.losses.softmax_cross_entropy()")
        cross_entropy = tf.losses.softmax_cross_entropy(
            logits=logits[0], onehot_labels=onehot_labels,
            label_smoothing=.5, reduction=tf.losses.Reduction.NONE) # tf.losses.Reduction.SUM
        loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')

    last_model = 'final_save.meta'

    with tf.Session() as sess:
        print('get_collection()')
        inception_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                           scope='Inception_Inference')
        print('len(get_collection()) returned: {}'.format(len(inception_vars)))
        print('initializing')
        _ = [v.initializer.run()for v in inception_vars]
        #sess.run(tf.global_variables_initializer())
        print('tf.train.import_meta_graph()')
        meta_graph_path = os.path.abspath(os.path.join(FLAGS.checkpoint_dir, last_model))
        #meta_graph_path = os.path.abspath(FLAGS.checkpoint_dir)
        print('meta_graph_path: {}'.format(meta_graph_path))
        saver = tf.train.import_meta_graph(meta_graph_path, clear_devices=True)
        print('saver.restore()')
        f = saver.restore(sess, tf.train.latest_checkpoint(FLAGS.checkpoint_dir))
        #print('initilize')
        #sess.run(tf.global_variables_initializer())
        print('tf.train.start_queue_runners(sess=sess)')
        tf.train.start_queue_runners(sess=sess)
        print('starting loop')
        for i in range(100):
            print('step {}, loss {}'.format(i, sess.run(loss)))

    return

if __name__ == "__main__":
    inception_imagenet_validate()    
