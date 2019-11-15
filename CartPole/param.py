import argparse
class Params():
    def __init__(self):
        return
    def get_pid_args(self):
        pid_params = argparse.ArgumentParser()
        pid_params.add_argument('--p_x', type = float, default = 24.56)
        pid_params.add_argument('--i_x', type = float, default = 0.0)
        pid_params.add_argument('--d_x', type = float, default = 3.34)
        pid_params.add_argument('--p_theta', type = float, default = 3.27)
        pid_params.add_argument('--i_theta', type = float, default = 0.0)
        pid_params.add_argument('--d_theta', type = float,  default = 20.88)
        pid_params.add_argument('--lamb', type = float, default = 0.0)
        pid_params.add_argument('--episode', type = int, default = 1)
        pid_params.add_argument('--step_limit', type = int, default = 10000)
        pid_params.add_argument('--isprint', type = int, default = 1)

        return pid_params.parse_args()

    def get_main_args(self):
        main_params = argparse.ArgumentParser()
        main_params.add_argument('--save_data_xlsx', type = str, default='input_data.xlsx',
                                help="The Path Save data xlsx")
        main_params.add_argument('--D_input_dimension', type = int, default = 5)
        main_params.add_argument('--G_input_dimension', type = int, default = 4)
        main_params.add_argument('--D_output_dimension', type = int, default = 1)
        main_params.add_argument('--G_output_dimension', type = int, default = 1)
        main_params.add_argument('--batch_size', type=int, default=128)
        main_params.add_argument('--pre_batch_size', type=int, default=10)

        return main_params.parse_args()