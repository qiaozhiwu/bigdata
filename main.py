from Fed import fed_train
from GRNN import grnn_lenet_attack

if __name__ == '__main__':
    command_str = "--dataset mnist --model LeNet --dp_mechanism Gaussian --dp_epsilon 15 --dp_clip 10 --lr_decay 1 --dp_composition Simple"
    # command_str = "--dataset mnist --model LeNet --dp_mechanism no_dp --dp_epsilon 15 --dp_clip 10 --lr_decay 1 --dp_composition Simple"
    # command_str = "--dataset mnist --model LeNet --dp_mechanism Laplace --dp_epsilon 15 --dp_clip 10 --lr_decay 1 --dp_composition Simple"
    fed_train(command_str)
    grnn_lenet_attack(command_str)
