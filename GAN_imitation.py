import numpy as np
import tensorflow as tf
import random
import network
from params import Param

args = Param().get_alg_args()


class Gan_imitation():
    def __init__(self, obs, action, real_obs, real_action):
        self.obs = obs
        self.action = action
        self.real_obs = real_obs
        self.real_action = real_action
        self.pre_batch_size = args.pre_batch_size
        self.batch_size = args.batch_size
        self.d_input_dimension = args.D_input_dimension
        self.d_output_dimension = args.D_output_dimension
        self.g_input_dimension = args.G_input_dimension
        self.learning_rate = args.learning_rate
        self.num_steps = args.num_steps

        self.creat_model()


    # input -- (none, 5)
    def discriminator(self, input):
        return network.mlp(x=input, scope="discriminator", output_dim=args.D_output_dimension)

    def generator(self, input):
        return network.mlp(x=input, scope="generator", output_dim=args.G_output_dimension)

    def creat_model(self):
        with tf.variable_scope("D_pre_train"):
            """
            pre trained the discriminator network to make the discriminator network better
            """
            self.pre_input = tf.placeholder(tf.float32, shape=(self.pre_batch_size, self.d_input_dimension))
            self.pre_label = tf.placeholder(tf.float32, shape=(self.pre_batch_size, self.d_output_dimension))

            self.D_pre = self.discriminator(self.pre_input)
            self.pre_loss = tf.reduce_mean(tf.square(self.pre_label - self.D_pre))
            learning_rate = tf.train.exponential_decay(self.learning_rate, 0, 100, 0.96, staircase=True)
            self.optimize_pre = tf.train.AdamOptimizer(learning_rate).minimize(self.pre_loss)


        with tf.variable_scope("Gen"):
            self.gen_input = tf.placeholder(tf.float32, shape=(self.batch_size, self.g_input_dimension))
            self.G = self.generator(self.gen_input)


        with tf.variable_scope("Dis") as scope:
            self.x = tf.placeholder(tf.float32, shape=(self.batch_size, self.d_input_dimension))
            self.D1 = self.discriminator(self.x)
            scope.reuse_variables()
            self.D2 = self.discriminator(tf.concat([self.gen_input, self.G]))

        # mean(log(D) + log(1-(D(G))))
        self.loss_d = tf.reduce_mean(-tf.log(self.D1) - tf.log(1 - self.D2))
        self.loss_g = tf.reduce_mean(-tf.log(self.D2))

        self.dis_pre_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Dis_pre") # trainable variables
        self.dis_param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Dis")
        self.g_param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Gen")

        self.optimize_dis = tf.train.AdamOptimizer().minimize(self.loss_d, var_list=self.dis_param)
        self.optimize_gen = tf.train.AdamOptimizer().minimize(self.loss_g, var_list=self.g_param)


    def train(self):
        # data for pretrain
        real_obs_pre_train = self.real_obs[0:self.pre_batch_size]
        real_action_pre_train = self.real_action[0:self.pre_batch_size]
        pre_input = np.hstack((real_obs_pre_train, real_action_pre_train))

        # data for train
        real_obs_train_data = self.real_obs[self.batch_size:self.batch_size*2]
        real_action_train_data = self.real_action[self.batch_size:self.batch_size*2]
        real_train_data = np.hstack((real_obs_train_data, real_action_train_data))
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for i in range(args.num_pretrain_step):
                # np.ones() -- the discriminator network should give the real data (expert actions) value 1,
                pre_loss, _ = sess.run([self.pre_loss, self.optimize_pre],
                         {self.pre_input:pre_input,
                          self.pre_label:np.ones((self.pre_batch_size, self.d_output_dimension))})

                # get the pre trained discriminator parameters
                self.dis_weights = sess.run(self.dis_pre_params)

                # copy

                for i, v in enumerate(self.dis_param):
                    sess.run(v.assign(self.dis_weights[i]))
                self.dis_weights = sess.run(self.dis_param)
                for step in range(self):
                    # update discriminator
                    dis_out = sess.run([self.D1], {self.x: real_train_data})
                    print("the output of the discriminator", dis_out)
                    loss_dis, _ = sess.run([self.loss_d, self.optimize_dis], {})

                    # update generator




            # after pre train the discriminator network,
            # we should copy the network parameters to the new discriminator network






