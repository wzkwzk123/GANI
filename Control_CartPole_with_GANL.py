import gym
env = gym.make('CartPole-v0')


def random_action():
    env.reset()
    for _ in range(1000):
        env.render()

        env.step(env.action_space.sample()) # take a random action
    env.close()

def control_GANL():
    # get obs from environment
    observation = env.reset()


    observation, reward, done, info = env.step(action)


def GANL_nwtwork():



if __name__ == "__main__":
    # random_action()
    # control_GANL()
    GANL_nwtwork()
