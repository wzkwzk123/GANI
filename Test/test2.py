import numpy as np
import tensorflow as tf

def main():
    p1 = tf.(tf.float32, name="param1")
    p2 = tf.placeholder(tf.float32, name="param2")

    p3 = tf.add(p1, p2)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(p3, {p1:2, p2:5})

    # result = v1 + v2

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.save(sess, "Model/model.ckpt")


if __name__ == "__main__":
    main()
