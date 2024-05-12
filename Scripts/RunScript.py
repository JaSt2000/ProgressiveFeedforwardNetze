import os
from os.path import join
import copy

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.backends.cudnn import benchmark
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import include.Model_and_Methods as mm
import include.strategies.strategy1_random_edges as strat1_random_edges
import include.strategies.strategy1_random_neurons as strat1_random_neurons
import include.strategies.strategy2_constant_edges as strat2_constant_edges
import include.strategies.strategy2_constant_neurons as strat2_constant_neurons
import include.strategies.strategy3_regularized_neurons as strat3_regularized_neurons
import include.strategies.strategy3_regularized_edges as strat3_regularized_edges
import include.strategies.strategy4_initial_edges as strat4_initial_edges
import include.strategies.strategy4_initial_neurons as strat4_initial_neurons
import include.strategies.strategy4_warmstarted_edges as strat4_warmstarted_edges
import include.strategies.strategy4_warmstarted_neurons as strat4_warmstarted_neurons
import include.strategies.strategy5_splitting_neurons as strat5_splitting_neurons
import include.strategies.strategy5_splitting_edges as strat5_splitting_edges
import include.strategies.strategy5_gradient_based_neurons as strat5_gradient_based_neurons
import include.strategies.strategy5_gradient_based_edges as strat5_gradient_based_edges
import include.strategies.strategy6_layer_statistics_edges as strat6_layerStat_edges
import include.strategies.strategy6_layer_statistics_neurons as strat6_layerStat_neurons


# for strategy 4
def warmstart_model(model, training_data, batch_size, warmstart_epochs=3):
    train_loader = torch.utils.data.DataLoader(dataset=training_data,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               generator=torch.Generator(device='cuda'))
    model.train()
    optimizer_warmstarted_model = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9,
                                                  nesterov=True)
    criterion_warmstarted_model = nn.CrossEntropyLoss()
    for ep in range(warmstart_epochs):
        # Loop over each batch from the training set
        for batch_idx, (data, target) in enumerate(train_loader):
            # Copy data to GPU if needed
            data = data.to(device)
            target = target.to(device)
            # Zero gradient buffers
            optimizer_warmstarted_model.zero_grad()
            # Pass data through the network
            output_layers = model(data)
            # Calculate loss
            loss = criterion_warmstarted_model(output_layers[-1], target)
            # Backpropagate - gradient computation
            loss.backward()
            # Update weights
            optimizer_warmstarted_model.step()
    model.eval()
    return


def run(seed, device, writer, training_data, test_data, num_total_expansions, eps, strategy_name, directory,
        expand_func, debug_grad_flag=False, learning_rate=0.001, gradient_clipping=100, initial_network_id=1):
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
    if "initial" in strategy_name:
        model_strategy4_initial = mm.ProgressiveMLP(n_layers=num_layers, n_neurons=num_neurons).to(device)
    elif "warmstarted" in strategy_name:
        # warmstart model strategy 4 variation
        model_strategy4_warmstarted = mm.ProgressiveMLP(n_layers=num_layers, n_neurons=num_neurons).to(device)
        warmstart_model(model_strategy4_warmstarted, training_data, batch_size)
        print("warmstart complete")

    state_dict = model.state_dict()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)  # , nesterov=True
    criterion = nn.CrossEntropyLoss()

    # define initial network through masks (initial network masks have value 1)
    if initial_network_id == 1:  # only 1 edge in the hidden layers
        for l in range(len(model.weight_mask_list) - 1):  # -1 to leave output layer unchanged (mask 1)
            model.weight_mask_list[l][1:,
            :] = 0
            model.bias_mask_list[l][1:] = 0
            if l != 0:
                model.weight_mask_list[l][0,
                1:] = 0
                model.bias_mask_list[l][0] = 0

            model.total_params_to_be_added[l] = torch.numel(model.weight_mask_list[l]) + torch.numel(
                model.bias_mask_list[l]) - torch.count_nonzero(model.weight_mask_list[l]) - torch.count_nonzero(
                model.bias_mask_list[l])  # Zahl der initial hinzuzufügenden Kanten pro Layer
            if l == 0:
                model.total_neurons_to_be_added[l] = model.weight_mask_list[l].size(
                    dim=0) - 1  # Zahl der initial hinzuzufügenden Neuronen pro Layer
            else:
                model.total_neurons_to_be_added[l] = model.weight_mask_list[l].size(
                    dim=0)  # da hier nur eine Kante jedes Neurons initialisiert ist

    elif initial_network_id == 2:  # only 1 neuron in the hidden layers
        for l in range(len(model.weight_mask_list) - 1):  # -1 to leave output layer unchanged (mask 1)
            model.weight_mask_list[l][1:,
            :] = 0
            model.bias_mask_list[l][1:] = 0

            model.total_params_to_be_added[l] = torch.numel(model.weight_mask_list[l]) + torch.numel(
                model.bias_mask_list[l]) - torch.count_nonzero(model.weight_mask_list[l]) - torch.count_nonzero(
                model.bias_mask_list[l])  # Zahl der initial hinzuzufügenden Kanten pro Layer
            model.total_neurons_to_be_added[l] = model.weight_mask_list[l].size(
                dim=0) - 1  # Zahl der initial hinzuzufügenden Neuronen pro Layer

    elif initial_network_id == 3:  # 3 neurons fully connected in the hidden layers
        for l in range(len(model.weight_mask_list) - 1):  # -1 to leave output layer unchanged (mask 1)
            model.weight_mask_list[l][3:,
            :] = 0
            model.bias_mask_list[l][3:] = 0
            if l != 0:  # disable all edges with no input from previous layer
                model.weight_mask_list[l][0:3,
                3:] = 0

            model.total_params_to_be_added[l] = torch.numel(model.weight_mask_list[l]) + torch.numel(
                model.bias_mask_list[l]) - torch.count_nonzero(model.weight_mask_list[l]) - torch.count_nonzero(
                model.bias_mask_list[l])  # Zahl der initial hinzuzufügenden Kanten pro Layer
            model.total_neurons_to_be_added[l] = model.weight_mask_list[l].size(
                dim=0) # Zahl der initial hinzuzufügenden Neuronen pro Layer (da ein Großteil der Kanten der 3 Neuronen noch mit 0 maskiert ist

    # set all network weights which are not part of the initial network to zero
    with torch.no_grad():
        for l in range(len(model.layers)):
            model.layers[l].weight *= model.weight_mask_list[l]
            model.layers[l].bias *= model.bias_mask_list[l]

    epoch = 1
    log_interval = 100
    num_expansions = 0
    epochs_since_last_expansion = 0

    epoch_vec, trn_lossv, trn_accv, tst_lossv, tst_accv, num_expansions_vec, num_params_vec, list_predecessor_index_expand, list_saved_weights_before_expansion, list_saved_masks, list_saved_weights_after_expansion, list_saved_masks_after_expansion = [], [], [], [], [], [], [], [], [], [], [], []

    while num_expansions < num_total_expansions or (epochs_since_last_expansion < 200 and np.abs(trn_lossv[-1] - trn_lossv[
                -1 - int(len(training_data) / (
                        batch_size * log_interval))]) > eps) or epochs_since_last_expansion < 10:   # python or & and are short circuit operators
        mm.train(device, writer, optimizer, model, train_loader, train_loader_eval, test_loader_eval, criterion, epoch,
                 tst_lossv, tst_accv,
                 trn_lossv, trn_accv, num_expansions_vec, num_params_vec, epoch_vec, log_interval, num_expansions,
                 debug_grad_flag, gradient_clipping)
        epochs_since_last_expansion += 1
        if epoch >= 2:
            if np.abs(trn_lossv[-1] - trn_lossv[
                -1 - int(len(training_data) / (
                        batch_size * log_interval))]) <= eps:  # compares current loss with loss entry recorded one epoch ago
                print("start of potential expand step")
                state_dict_before_expansion = copy.deepcopy(model.state_dict())
                mask_dict_before_expansion = {"weights": copy.deepcopy(model.weight_mask_list),
                                              "biases": copy.deepcopy(model.bias_mask_list)}
                if ("regularized" in strategy_name) or ("splitting_edges" in strategy_name):
                    expanded = expand_func(device, model, num_total_expansions,
                                           training_data,
                                           num_expansions)  # convey , training_data for strategy3/strategy5
                elif "splitting_neurons" in strategy_name:
                    # print("number of nonzero params: ", model.non_zero_params_count())
                    if "eigenvectors" in strategy_name:
                        expanded = expand_func(device, model, num_total_expansions,
                                               training_data,
                                               False,
                                               num_expansions)  # convey training_data and random_init_flag for strategy5 (false means eigenvectors)
                    else:
                        expanded = expand_func(device, model, num_total_expansions,
                                               training_data,
                                               True,
                                               num_expansions)  # convey training_data and random_init_flag for strategy5 (true means random)
                    # print(list(model.parameters()))
                elif "initial" in strategy_name:
                    expanded = expand_func(device, model, num_total_expansions,
                                           model_strategy4_initial,
                                           num_expansions)
                elif "warmstarted" in strategy_name:
                    expanded = expand_func(device, model, num_total_expansions,
                                           model_strategy4_warmstarted,
                                           num_expansions)
                elif "layer_stat" in strategy_name:
                    expanded = expand_func(device, model, num_total_expansions,
                                           training_data,
                                           True,
                                           num_expansions)  # convey training_data and use_variance_values_flag for strategy6 (false utilizes only the rank in the sorted list)
                elif "gradient" in strategy_name:
                    expanded = expand_func(device, model, num_total_expansions,
                                           training_data,
                                           True,
                                           num_expansions)  # convey training_data and use_gradient_values_flag for strategy5 (false utilizes only the rank in the sorted list)
                else:
                    expanded = expand_func(device, model, num_total_expansions,
                                           num_expansions)  # num_expansions: number of expansions unil now (not counting the current one)

                if expanded:
                    state_dict_after_expansion = copy.deepcopy(model.state_dict())
                    mask_dict_after_expansion = {"weights": copy.deepcopy(model.weight_mask_list),
                                                 "biases": copy.deepcopy(model.bias_mask_list)}
                    list_saved_masks_after_expansion.append(mask_dict_after_expansion)
                    list_saved_weights_after_expansion.append(state_dict_after_expansion)
                    list_saved_weights_before_expansion.append(state_dict_before_expansion)
                    list_saved_masks.append(mask_dict_before_expansion)
                    epochs_since_last_expansion = 0
                    num_expansions += 1
                    # mark current loss entry as immediate predecessor of an expand step
                    list_predecessor_index_expand.append(len(tst_lossv) - 1)
                print("end of expand step. Actual expansion: ", expanded)
        epoch += 1
    list_saved_weights_before_expansion.append(copy.deepcopy(model.state_dict()))
    result_dict = {"train_loss": trn_lossv, "train_accuracy": trn_accv, "test_loss": tst_lossv,
                   "test_accuracy": tst_accv, "num_expansions": num_expansions_vec, "num_params": num_params_vec,
                   "index_before_expand": list_predecessor_index_expand, "epoch": epoch_vec,
                   "weights": list_saved_weights_before_expansion,          # weights before every expansion (plus a final state)
                   "masks": list_saved_masks,
                   "weights_after_expand": list_saved_weights_after_expansion,
                   "masks_after_expand": list_saved_masks_after_expansion}
    torch.save(result_dict,
               directory + "/" + strategy_name + f"_result_seed_{seed}_expansions_{num_total_expansions}_eps_{eps}_lr_{learning_rate}_grad_clip_{gradient_clipping}_initial_network_{initial_network_id}.pt")


if __name__ == '__main__':
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    # set device and seed and summaryWriter for tensorboard
    # writer = SummaryWriter()
    device = (
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )
    torch.set_default_device(device)
    torch.backends.cudnn.benchmark = False
    # torch.backends.cuda.preferred_linalg_library("magma")
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
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = mnist_dataloader.load_data()

    # convert to tensor
    X_train_mnist = torch.Tensor(np.array(x_train_mnist)).to(device)
    y_train_mnist = torch.Tensor(y_train_mnist).to(torch.long).to(device)
    X_test_mnist = torch.Tensor(np.array(x_test_mnist)).to(device)
    y_test_mnist = torch.Tensor(y_test_mnist).to(torch.long).to(device)

    # tensorboard --logdir=C:\Users\janst\PycharmProjects\BachelorArbeit\Scripts\runs
    # dataloaders & batchsize
    training_set_mnist = mm.PytorchDataset(X_train_mnist, y_train_mnist)
    test_set_mnist = mm.PytorchDataset(X_test_mnist, y_test_mnist)

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0,), (1 / 255,)), ])

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

    # print(training_set_fashion_mnist.x.dtype)
    # print(training_set_fashion_mnist.y.dtype)

    dataset_name_mnist = "MNIST"
    dataset_name_fashion_mnist = "Fashion_MNIST"

    # current_dataset = dataset_name_fashion_mnist
    # current_training_set = training_set_fashion_mnist
    # current_test_set = test_set_fashion_mnist
    current_dataset = dataset_name_fashion_mnist
    current_training_set = training_set_fashion_mnist
    current_test_set = test_set_fashion_mnist

    num_total_expansions = 5
    eps = 0.01
    lr = 0.001
    grad_clip = 100

    # strategy specifics

    directories = dict()
    expand_funcs = dict()

    directories["random_edges"] = "../results/strategy1_random/edges" + "/" + current_dataset
    expand_funcs["random_edges"] = strat1_random_edges.random_edges_expand

    directories["random_neurons"] = "../results/strategy1_random/neurons" + "/" + current_dataset
    expand_funcs["random_neurons"] = strat1_random_neurons.random_neurons_expand

    directories["constant_edges"] = "../results/strategy2_constant/edges" + "/" + current_dataset
    expand_funcs["constant_edges"] = strat2_constant_edges.constant_edges_expand

    directories["constant_neurons"] = "../results/strategy2_constant/neurons" + "/" + current_dataset
    expand_funcs["constant_neurons"] = strat2_constant_neurons.constant_neurons_expand

    directories["regularized_neurons"] = "../results/strategy3_regularized/neurons" + "/" + current_dataset
    expand_funcs["regularized_neurons"] = strat3_regularized_neurons.regularized_neurons_expand

    directories["regularized_edges"] = "../results/strategy3_regularized/edges" + "/" + current_dataset
    expand_funcs["regularized_edges"] = strat3_regularized_edges.regularized_edges_expand

    directories["initial_edges"] = "../results/strategy4_initial/edges" + "/" + current_dataset
    expand_funcs["initial_edges"] = strat4_initial_edges.initial_edges_expand

    directories["initial_neurons"] = "../results/strategy4_initial/neurons" + "/" + current_dataset
    expand_funcs["initial_neurons"] = strat4_initial_neurons.initial_neurons_expand

    directories["warmstarted_edges"] = "../results/strategy4_warmstarted/edges" + "/" + current_dataset
    expand_funcs["warmstarted_edges"] = strat4_warmstarted_edges.warmstarted_edges_expand

    directories["warmstarted_neurons"] = "../results/strategy4_warmstarted/neurons" + "/" + current_dataset
    expand_funcs["warmstarted_neurons"] = strat4_warmstarted_neurons.warmstarted_neurons_expand

    directories["gradient_based_edges"] = "../results/strategy5_gradient_based/edges" + "/" + current_dataset
    expand_funcs["gradient_based_edges"] = strat5_gradient_based_edges.gradient_based_edges_expand

    directories["gradient_based_neurons"] = "../results/strategy5_gradient_based/neurons" + "/" + current_dataset
    expand_funcs["gradient_based_neurons"] = strat5_gradient_based_neurons.gradient_based_neurons_expand

    directories["layer_stat_edges"] = "../results/strategy6_layer_stat/edges" + "/" + current_dataset
    expand_funcs["layer_stat_edges"] = strat6_layerStat_edges.layer_stat_edges_expand

    directories["layer_stat_neurons"] = "../results/strategy6_layer_stat/neurons" + "/" + current_dataset
    expand_funcs["layer_stat_neurons"] = strat6_layerStat_neurons.layer_stat_neurons_expand

    # directories["splitting_neurons"] = "../results/strategy5_splitting/neurons" + "/" + current_dataset
    # expand_funcs[
    #     "splitting_neurons"] = strat5_splitting_neurons.splitting_neurons_expand  # strat5_splitting_neurons.splitting_neurons_expand
    #
    # directories[
    #     "splitting_neurons_eigenvectors_init"] = "../results/strategy5_splitting/neurons" + "/" + current_dataset
    # expand_funcs[
    #     "splitting_neurons_eigenvectors_init"] = strat5_splitting_neurons.splitting_neurons_expand  # strat5_splitting_neurons.splitting_neurons_expand
    #
    # directories["splitting_edges"] = "../results/strategy5_splitting/edges" + "/" + current_dataset
    # expand_funcs["splitting_edges"] = strat5_splitting_edges.splitting_edges_expand

    #
    # writer = SummaryWriter(
    #     log_dir="runs/" + strategy_name_splitting_neurons_eigenvectors_init + f"eigenvectors_seed_{1}_num_expansions_{num_total_expansions}_eps_{eps}")
    # run(1, device, writer, training_data, test_data, num_total_expansions, eps, strategy_name_splitting_neurons_eigenvectors_init,
    #     directory_splitting_neurons, expand_func_splitting_neurons)

    for seed in range(4, 6):
        for strategy_name in directories.keys():
            for initial_network_id in range(1, 4):
                # if ("gradient_based_edges" in strategy_name): #or ("layer_stat" in strategy_name):   #TODO: Layer_stat_seed 3 & all strategies seed 4+5
                # if "layer_stat" in strategy_name:
                writer = SummaryWriter(
                    log_dir="runs/" + current_dataset + "/" + strategy_name + f"_seed_{seed}_num_expansions_{num_total_expansions}_eps_{eps}_initial_network_{initial_network_id}")
                run(seed, device, writer, current_training_set, current_test_set, num_total_expansions, eps,
                    strategy_name,
                    directories[strategy_name], expand_funcs[strategy_name], False, lr, grad_clip, initial_network_id)
                print("------------- seed ", seed, " done-------------------------------------------")
