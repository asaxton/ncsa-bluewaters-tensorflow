# NCSA Blue Waters Tensorflow
This repo contains scripts and utilities to run Distributed Tensorflow jobs on
Blue Waters.

The `datasets` directory have scripts and tools that will processes raw
data into a Tensorflow native TFrecord format. If you are lucky, there will be
a script there that you can use right out of the box. Otherwise, use the tools
there as template and examples that you can build off of.

The `run_scripts` directory will contain scripts and tool that will actually run
your distributed Tensorflow app
