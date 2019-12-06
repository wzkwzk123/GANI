import gym
import tensorflow as tf
env = gym.make('CartPole-v0')


def random_action():
    env.reset()
    for _ in range(1000):
        env.render()
        print(env.action_space.sample())

        env.step(env.action_space.sample()) # take a random action
    env.close()

def control_GANL():
    # get obs from environment
    observation = env.reset()
    observation, reward, done, info = env.step(action)


def GANL_nwtwork():
    sess = tf.Session()
    saver = tf.train.import_meta_graph("./Model/model.ckpt-2000.meta")
    # saver.restore(sess, tf.train.latest_checkpoint('./Model'))
    saver.restore(sess, "./Model/model.ckpt-2000")
    graph = tf.get_default_graph()
    g_input_ = graph.get_tensor_by_name("Gen/g_input:0")
    g_output = graph.get_tensor_by_name("Gen/generator/output_layer/Sigmoid:0")

    observation = env.reset()
    print("observation we get from environment", observation)

    for _ in range(1000):
        env.render()
        action_GANL = sess.run(g_output, feed_dict={g_input_:[observation]})
        print("the output from generator network:", action_GANL)
        print("the action we will take:", int(round(action_GANL[0][0])))
        observation, reward, done, info = env.step(int(round(action_GANL[0][0])))
        print("observation we get from environment", observation)
    env.close()


if __name__ == "__main__":
    # random_action()
    GANL_nwtwork()
