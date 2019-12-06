import numpy as np
import tensorflow as tf
import random
import network
from params import Param
from DataProcess import data_save_read

args = Param().get_alg_args()


class Gan_imitation():
    def __init__(self, real_obs, real_action, log_fre, args):

        self.real_obs = real_obs
        self.real_action = real_action
        self.pre_batch_size = args.pre_batch_size
        self.batch_size = args.batch_size
        self.d_input_dimension = args.D_input_dimension
        self.d_output_dimension = args.D_output_dimension
        self.g_input_dimension = args.G_input_dimension
        self.learning_rate = args.learning_rate
        self.num_steps = args.num_steps
        self.log_fre = 100
        self.save_flag = args.save_flag

        self.creat_model()


    # input -- (none, 5)
    def discriminator(self, input):
        return network.mlp(x=input,hidden_sizes=30, scope="discriminator", output_dim=args.D_output_dimension)

    def generator(self, input):
        return network.mlp(x=input, hidden_sizes=30, scope="generator", output_dim=args.G_output_dimension)

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
            print("learning_rate", learning_rate)
            self.optimize_pre = tf.train.AdamOptimizer(learning_rate).minimize(self.pre_loss)


        with tf.variable_scope("Gen"):
            self.gen_input = tf.placeholder(tf.float32, shape=(None, self.g_input_dimension), name="g_input")
            self.G = self.generator(self.gen_input)


        with tf.variable_scope("Dis") as scope:
            self.x = tf.placeholder(tf.float32, shape=(None, self.d_input_dimension))
            self.D1 = self.discriminator(self.x)
            scope.reuse_variables()
            self.D2 = self.discriminator(tf.concat([self.gen_input, self.G], 1))

        # mean(log(D) + log(1-(D(G))))
        # the inputs for self.loss_d contains both expert and generator network output data
        # small_number = 0.000000001
        self.loss_d = tf.reduce_mean(-tf.log(self.D1) - tf.log(1 - self.D2))
        self.loss_g = tf.reduce_mean(-tf.log(self.D2))

        self.dis_pre_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="D_pre_train") # trainable variables
        self.dis_param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Dis")
        self.g_param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Gen")

        learning_rate = tf.train.exponential_decay(self.learning_rate, 0, 100, 0.96, staircase=True)
        self.optimize_dis = tf.train.AdamOptimizer(0.00001).minimize(self.loss_d, var_list=self.dis_param)
        self.optimize_gen = tf.train.AdamOptimizer(0.00001).minimize(self.loss_g, var_list=self.g_param)


    def train(self):
        # data for pretrain
        real_obs_pre_train = self.real_obs[0:self.pre_batch_size]
        real_action_pre_train = self.real_action[0:self.pre_batch_size]
        pre_input = np.hstack((real_obs_pre_train, real_action_pre_train))

        # data for train
        real_obs_train_data = self.real_obs[0:self.batch_size*3]
        real_action_train_data = self.real_action[0:self.batch_size*3]
        real_train_data = np.hstack((real_obs_train_data, real_action_train_data))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for i in range(args.num_pretrain_step):
                # np.ones() -- the discriminator network should give the real data (expert actions) value 1,
                pre_loss, _ = sess.run([self.pre_loss, self.optimize_pre],
                         {self.pre_input:np.reshape(pre_input, (self.pre_batch_size, self.d_input_dimension)),
                          self.pre_label:np.ones((self.pre_batch_size, self.d_output_dimension))})
                print("pre_loss", pre_loss)

                # get the pre trained discriminator parameters
            self.dis_weights = sess.run(self.dis_pre_params)

            # copy

            for i, v in enumerate(self.dis_param):
                sess.run(v.assign(self.dis_weights[i]))
            self.dis_weights = sess.run(self.dis_param)

            for step in range(self.num_steps):
                # update discriminator
                dis_out = sess.run([self.D1], {self.x: real_train_data})
                # print("the output of the discriminator", dis_out)
                for jjj in range(30):
                    loss_dis, _ = sess.run([self.loss_d, self.optimize_dis],
                                               {self.x:real_train_data,
                                                self.gen_input:real_obs_train_data
                                                })
                # update generator
                for jjj in range(30):
                    loss_g, _ = sess.run([self.loss_g, self.optimize_gen], {self.gen_input:real_obs_train_data})

                if step % self.log_fre == 0:
                    print('{}:loss_dis:{}\tloss_g:{}'.format(step, loss_dis, loss_g))

                if self.save_flag:
                    if step % 500 == 0:
                        saver = tf.train.Saver()
                        saver.save(sess,"Model/model.ckpt", global_step=step) # specify sess instance

            if True:
                # tensor board
                writer = tf.summary.FileWriter("logs/", tf.get_default_graph())



if __name__ == "__main__":
    data = data_save_read.read_data('input_data.xlsx')
    random.shuffle(data)
    data_obs = []
    data_label = []
    argsmain = Param().get_alg_args()
    for i in range(len(data)):
        # Get Train Data And Label
        # print('i {}, data {}, label {}'.format(i, data[i][0:-1], data[i][-1:]))
        data_obs.append(data[i][0:-1])
        data_label.append(data[i][-1:])
    model = Gan_imitation(data_obs, data_label, 10, argsmain)
    model.train()



