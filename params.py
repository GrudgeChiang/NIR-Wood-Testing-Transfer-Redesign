batch_size = 512
use_gpu = True
data_root = './data'
dataset_mean = (0.5, 0.5, 0.5)
dataset_std = (0.5, 0.5, 0.5)
mnist_path = data_root + '/MNIST'
mnistm_path = data_root + '/MNIST_M'
epochs = 1
plot_iter = 10
lr = 0.001
momentum = 0.9

theta1 = 0.5
theta2 = 0.5
theta3 = 0.1

N_GEN_EPOCHS = 1
N_CDAN_EPOCHS = 1
N_ETD_CDAN_EPOCHS = 1
N_ETD_EPOCHS = 10