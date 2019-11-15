import gym

from param import Params
import CartPoleControl

argspid = Params().get_pid_args()

fitness = CartPoleControl.CratpoleControl(argspid)

