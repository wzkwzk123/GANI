import gym
import matplotlib.pyplot as plt
from DataProcess import data_save_read
env = gym.make('CartPole-v1')
import sys
sys.path.append("./../")
from params import Param

args = Param().get_pid_args()

def CratpoleControl(args):

    for i_episode in range(args.episode):
        observation = env.reset()
        v_x = 0
        x_t = x_t1 = x_t2 = 0
        v_theta = 0
        theta_t = theta_t1 = theta_t2 = 0
        time_step = []
        ob_plot1 = []
        ob_plot2 = []
        ob_plot3 = []
        ob_plot4 = []
        action_plot = []
        for t in range(args.step_limit):
            ob_plot1.append(observation[0])
            ob_plot2.append(observation[1])
            ob_plot3.append(observation[2])
            ob_plot4.append(observation[3])
            time_step.append(t)
            env.render()
            v_x = v_x + args.p_x * (x_t - x_t1) + args.d_x * (x_t - 2 * x_t1 + x_t2) + args.i_x * x_t
            v_theta = v_theta + args.p_theta * (theta_t - theta_t1) + args.d_theta * (
                    theta_t - 2 * theta_t1 + theta_t2) + args.i_theta * theta_t
            v = args.lamb * v_x + (1 - args.lamb) * v_theta
            if v > 0:
                action = 1
            else:
                action = 0
            action_plot.append(action)
            # state = (x,x_dot,theta,theta_dot)
            observation, reward, done, info = env.step(action)
            x_t2 = x_t1
            x_t1 = x_t
            x_t = observation[0]
            theta_t2 = theta_t1
            theta_t1 = theta_t
            theta_t = x_t = observation[2]
            if done:
                if args.isprint:
                    print("Episode finished after {} timesteps".format(t + 1))
                data_save_read.write_date(ob_plot1, ob_plot2, ob_plot3, ob_plot4, action_plot)
                # plot_figure(time_step, ob_plot1, ob_plot2, ob_plot3, ob_plot4)
                break
        if args.isprint and t == args.step_limit:
            print("Episode finished after {} timesteps".format(t + 1))

        # data_save_read.write_date(ob_plot1, ob_plot2, ob_plot3, ob_plot4, action_plot)
    return True

def plot_figure(time_step, ob_plot1, ob_plot2, ob_plot3, ob_plot4):
    plt.subplot(221)
    plt.plot(time_step, ob_plot1)
    plt.title('X')
    plt.subplot(222)
    plt.plot(time_step, ob_plot2)
    plt.title('X_dot')
    plt.subplot(223)
    plt.plot(time_step, ob_plot3)
    plt.title('theta')
    plt.subplot(224)
    plt.plot(time_step, ob_plot4)
    plt.title('theta_dot')
    plt.show()

if __name__ == "__main__":
    CratpoleControl(args)
