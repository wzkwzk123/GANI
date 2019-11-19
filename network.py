import tensorflow as tf
import numpy as np


def mlp(x, hidden_sizes, output_dim, scope):
    with tf.variable_scope(scope):
        output_1 = tf.layers.dense(x, units=hidden_sizes, activation=tf.nn.relu, name="fisrt_layer")
        out_put2 = tf.layers.dense(output_1, units=30, activation=tf.nn.relu, name="second_layer")
        out_put3 = tf.layers.dense(out_put2, units=output_dim, activation=tf.nn.sigmoid, name="output_layer")
    return out_put3



