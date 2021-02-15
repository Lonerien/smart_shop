# -*- coding: utf-8 -*-

import os
import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_file', '', 'Dataset .data file')
tf.app.flags.DEFINE_string('model_cfg', '', 'Model .cfg file')
tf.app.flags.DEFINE_string('start_weights', '', 'Weighs to start with, e.g. darknet.conv.74')

def main(argv=None):
    os.chdir('./darknet')
    os.system(f'./darknet detector train {FLAGS.data_file} {FLAGS.model_cfg} {FLAGS.start_weights}')

if __name__ == '__main__':
    tf.app.run()
