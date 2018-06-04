The general workflow is, train a model, validate a mode, deploy a model. The
scripts are boiler plate examples to train and validate.

inception_imagenet_distributed_train.pbs is a boiler plate script that runs the distributed
Tensorflow app inception_imagenet_distributed_train.py. Before running this script you should
read then run datasets/imagenet/extract_data_from_archive.pbs then
datasets/imagenet/build_imagenet_data.pbs

The checkpoint output is the result of training. If your job was terminated
before training completed, TensorFlow did not save the checkpoint correctly
for you to use it later. The script restore_graph.pbs  is a boiler plate
example about how to repair the checkpoint so that it can be either validated
or deployed.

Validating a model is usually the least computational task is working with
machine learning models. However it is often convenient to validate with the
same compute architecture that training was run on. The script
inception_imagenet_validate.pbs is a boiler plate example for this task.

Cray has provided us with a machine learning plugin that allows TensorFlow to
perform distributed training with out TensorFlows native
ParameterServer-Worker architecture. The scrip
inception_imagenet_distributed_cray_train.pbs is a boiler plate example on how
to run TensorFlow with Cray’s ML Plugin

The scripts in this directory are only the “Top of the Stack”. Most the actual
work is done in the python scripts located in BWDistributedTrain. Be sure to
familiarize yourself with the code located there.
