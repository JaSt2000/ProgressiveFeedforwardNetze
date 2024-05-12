import os
from os.path import join

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.backends.cudnn import benchmark
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import include.Model_and_Methods as mm


def run(seed, device, writer, training_data, test_data, dataset_name="MNIST", debug_grad_flag=False, lr=0.001, gradient_clipping=100, terminate_when_converged=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    batch_size = 100

    train_loader = torch.utils.data.DataLoader(dataset=training_data,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               generator=torch.Generator(device='cuda'))

    test_loader_eval = torch.utils.data.DataLoader(dataset=test_data,
                                                   batch_size=batch_size,
                                                   shuffle=False,
                                                   generator=torch.Generator(
                                                       device='cuda'))  # generator=torch.Generator(device='cuda')
    train_loader_eval = torch.utils.data.DataLoader(dataset=training_data,
                                                    batch_size=batch_size,
                                                    shuffle=False,
                                                    generator=torch.Generator(device='cuda'))

    # model
    num_layers = 6
    num_neurons = 30
    model = mm.ProgressiveMLP(n_layers=num_layers, n_neurons=num_neurons).to(device)
    state_dict = model.state_dict()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)  #, nesterov=True
    criterion = nn.CrossEntropyLoss()

    eps = 0.01 # threshold that decides loss convergence (only relevant if terminate_when_converged=True)

    epochs = 50
    log_interval = 100

    epoch_vec, trn_lossv, trn_accv, tst_lossv, tst_accv, num_expansions_vec, num_params_vec = [], [], [], [], [], [], []

    for epoch in range(1, epochs + 1):
        mm.train(device, writer, optimizer, model, train_loader, train_loader_eval, test_loader_eval, criterion, epoch,
                 tst_lossv, tst_accv,
                 trn_lossv, trn_accv, num_expansions_vec, num_params_vec, epoch_vec,
                 log_interval, 0, debug_grad_flag, gradient_clipping)
        if epoch >= 2 and terminate_when_converged and np.abs(trn_lossv[-1] - trn_lossv[-1 - int(len(training_data) / (batch_size * log_interval))]) <= eps:
            break
    result_dict = {"train_loss": trn_lossv, "train_accuracy": trn_accv, "test_loss": tst_lossv,
                   "test_accuracy": tst_accv, "epoch": epoch_vec}
    added_name=""
    if terminate_when_converged:
        added_name = "_with_loss_convergence_termination"
    torch.save(result_dict, f"../results/baseline/" + dataset_name + f"/baseline{added_name}_result_seed_{seed}_lr_{lr}_grad_clip_{gradient_clipping}.pt")


if __name__ == '__main__':
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    device = (
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )
    torch.set_default_device(device)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    print('Using PyTorch version:', torch.__version__, ' Device:', device)

    #
    # Load MINST dataset
    #
    input_path = '../MNIST/'
    training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
    training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
    test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
    test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')
    mnist_dataloader = mm.MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath,
                                          test_labels_filepath)
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

    # convert to tensor
    X_train = torch.Tensor(np.array(x_train)).to(device)
    y_train = torch.Tensor(y_train).to(torch.long).to(device)
    X_test = torch.Tensor(np.array(x_test)).to(device)
    y_test = torch.Tensor(y_test).to(torch.long).to(device)

    # tensorboard --logdir=absolutePathToRuns
    # dataloaders & batchsize
    training_set_mnist = mm.PytorchDataset(X_train, y_train)
    test_set_mnist = mm.PytorchDataset(X_test, y_test)

    # fashion mnist
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0,), (1 / 255,)), ])  # transforms.ConvertImageDtype(torch.int8)

    # Create datasets for training & validation, download if necessary
    training_set_fashion_mnist = torchvision.datasets.FashionMNIST('../Fashion_MNIST', train=True, transform=transform,
                                                                   download=True)
    test_set_fashion_mnist = torchvision.datasets.FashionMNIST('../Fashion_MNIST', train=False, transform=transform,
                                                               download=True)

    training_set_fashion_mnist.data = training_set_fashion_mnist.data.to(device).to(torch.float32)
    training_set_fashion_mnist.targets = training_set_fashion_mnist.targets.to(device).to(torch.long)

    test_set_fashion_mnist.data = test_set_fashion_mnist.data.to(device).to(torch.float32)
    test_set_fashion_mnist.targets = test_set_fashion_mnist.targets.to(device).to(torch.long)

    training_set_fashion_mnist = mm.PytorchDataset(training_set_fashion_mnist.data, training_set_fashion_mnist.targets)
    test_set_fashion_mnist = mm.PytorchDataset(test_set_fashion_mnist.data, test_set_fashion_mnist.targets)

    dataset_name_mnist = "MNIST"
    dataset_name_fashion_mnist = "Fashion_MNIST"

    current_dataset = dataset_name_fashion_mnist
    current_training_set = training_set_fashion_mnist
    current_test_set = test_set_fashion_mnist
    lr = 0.001
    grad_clip = 100

    # execute run
    for seed in range(1, 6):
        writer = SummaryWriter(
            log_dir="runs/" + dataset_name_mnist + f"/baseline_with_termination_on_convergence_seed_{seed}_grad_clip{grad_clip}")
        run(seed, device, writer, training_set_mnist, test_set_mnist, dataset_name_mnist, False, lr, grad_clip, terminate_when_converged=True)
        writer = SummaryWriter(
            log_dir="runs/" + dataset_name_fashion_mnist + f"/baseline_with_termination_on_convergence_seed_{seed}_grad_clip{grad_clip}")
        run(seed, device, writer, training_set_fashion_mnist, test_set_fashion_mnist, dataset_name_fashion_mnist, False, lr, grad_clip, terminate_when_converged=True)

