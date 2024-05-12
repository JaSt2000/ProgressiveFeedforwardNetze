import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import include.Model_and_Methods as mm
import os
from os.path import join
from torch.utils.data import DataLoader
import pandas as pd

device = (
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )
torch.set_default_device(device)

def load_result_dicts(directory, strategy_name, num_seeds, num_total_expansions, eps, lr, grad_clip, initial_network_id):
    result_dict_list = []
    for seed in range(1, num_seeds + 1):
        if num_total_expansions > 0:
            result_dict_list.append(torch.load(
                directory + "/" + strategy_name + f"_result_seed_{seed}_expansions_{num_total_expansions}_eps_{eps}_lr_{lr}_grad_clip_{grad_clip}_initial_network_{initial_network_id}.pt"))
        else:
            result_dict_list.append(torch.load(
                directory + "/" + strategy_name + f"_result_seed_{seed}_lr_{lr}_grad_clip_{grad_clip}.pt"))
    return result_dict_list

def get_result_list_end_epoch(list, training_data_size=60000 ,log_interval=100, batch_size=100):  # does not hold for "index_before_expand"
    stepsize = int(training_data_size / (log_interval * batch_size))
    return list[stepsize - 1::stepsize]


if __name__=="__main__":
    num_expansions = 5
    eps = 0.01
    num_seeds = 5
    initial_network_id = 3
    lr = 0.001
    grad_clip = 100
    fashion_mnist_name = "Fashion_MNIST"
    mnist_name = "MNIST"
    dataset_name = fashion_mnist_name
    strategies_result_dict = dict()
    strategies_result_dict["random_edges"] = load_result_dicts("../results/strategy1_random/edges" + "/" + dataset_name,
                                                               "random_edges", num_seeds, num_expansions, eps, lr, grad_clip, initial_network_id)
    strategies_result_dict["random_neurons"] = load_result_dicts(
        "../results/strategy1_random/neurons" + "/" + dataset_name, "random_neurons", num_seeds, num_expansions,
        eps, lr, grad_clip, initial_network_id)
    strategies_result_dict["constant_edges"] = load_result_dicts(
        "../results/strategy2_constant/edges" + "/" + dataset_name, "constant_edges", num_seeds, num_expansions,
        eps, lr, grad_clip, initial_network_id)
    strategies_result_dict["constant_neurons"] = load_result_dicts(
        "../results/strategy2_constant/neurons" + "/" + dataset_name, "constant_neurons", num_seeds,
        num_expansions, eps, lr, grad_clip, initial_network_id)
    strategies_result_dict["regularized_edges"] = load_result_dicts(
        "../results/strategy3_regularized/edges" + "/" + dataset_name, "regularized_edges",
        num_seeds, num_expansions,
        eps, lr, grad_clip, initial_network_id)
    strategies_result_dict["regularized_neurons"] = load_result_dicts(
        "../results/strategy3_regularized/neurons" + "/" + dataset_name,
        "regularized_neurons", num_seeds,
        num_expansions, eps, lr, grad_clip, initial_network_id)
    strategies_result_dict["initial_edges"] = load_result_dicts(
        "../results/strategy4_initial/edges" + "/" + dataset_name, "initial_edges", num_seeds, num_expansions, eps, lr, grad_clip, initial_network_id)
    strategies_result_dict["initial_neurons"] = load_result_dicts(
        "../results/strategy4_initial/neurons" + "/" + dataset_name, "initial_neurons", num_seeds, num_expansions, eps, lr, grad_clip, initial_network_id)

    strategies_result_dict["warmstarted_edges"] = load_result_dicts(
        "../results/strategy4_warmstarted/edges" + "/" + dataset_name, "warmstarted_edges", num_seeds, num_expansions,
        eps,
        lr, grad_clip, initial_network_id)
    strategies_result_dict["warmstarted_neurons"] = load_result_dicts(
        "../results/strategy4_warmstarted/neurons" + "/" + dataset_name, "warmstarted_neurons", num_seeds,
        num_expansions,
        eps, lr, grad_clip, initial_network_id)

    strategies_result_dict["gradient_based_edges"] = load_result_dicts(
        "../results/strategy5_gradient_based/edges" + "/" + dataset_name, "gradient_based_edges", num_seeds,
        num_expansions, eps, lr, grad_clip, initial_network_id)
    strategies_result_dict["gradient_based_neurons"] = load_result_dicts(
        "../results/strategy5_gradient_based/neurons" + "/" + dataset_name, "gradient_based_neurons", num_seeds,
        num_expansions, eps, lr, grad_clip, initial_network_id)

    strategies_result_dict["layer_stat_edges"] = load_result_dicts(
        "../results/strategy6_layer_stat/edges" + "/" + dataset_name, "layer_stat_edges", num_seeds, num_expansions,
        eps, lr, grad_clip, initial_network_id)
    strategies_result_dict["layer_stat_neurons"] = load_result_dicts(
        "../results/strategy6_layer_stat/neurons" + "/" + dataset_name, "layer_stat_neurons", num_seeds, num_expansions,
        eps, lr, grad_clip, initial_network_id)

    # strategies_result_dict["splitting_neurons_eigenvectors_init"] = load_result_dicts("../results/strategy5_splitting/neurons" + "/" + dataset_name, "splitting_neurons_eigenvectors_init", num_seeds, num_expansions, eps)

    # result_dict_list_layer_statistics_edges = load_result_dicts("../results/strategy6_layer_statistics/edges", "layer_statistics_edges", num_seeds, num_expansions, eps)
    # result_dict_list_layer_statistics_neurons = load_result_dicts("../results/strategy6_layer_statistics/neurons", "layer_statistics_neurons", num_seeds, num_expansions, eps)

    strategies_result_dict["baseline"] = load_result_dicts("../results/baseline" + "/" + dataset_name, "baseline",
                                                           num_seeds, 0, 0, lr, grad_clip, 0)

    debug_flag = False
    progress_flag = True
    strategies_connectivity_dict = dict()
    for strategy in strategies_result_dict:
        if strategy != "baseline":
            strategies_connectivity_dict[strategy] = []

    for strategy in strategies_connectivity_dict:
        for expansion in range(num_expansions):
            mean_connectivity_score = 0
            for seed in range(num_seeds):
                if progress_flag:
                    print("----------------------- Strategy: ", strategy, ", Expansion: ", expansion, " ,Seed: ", seed,
                          " -----------------------")
                weight_mask_list = strategies_result_dict[strategy][seed]["masks"][expansion]["weights"]
                mean_connectivity_score += mm.get_connectivity_score(weight_mask_list, debug_flag=debug_flag)
            mean_connectivity_score /= num_seeds
            strategies_connectivity_dict[strategy].append(mean_connectivity_score)
        mean_last_connectivity_score = 0
        for s in range(num_seeds):
            weight_mask_list_last = strategies_result_dict[strategy][s]["masks_after_expand"][4]["weights"]
            mean_last_connectivity_score += mm.get_connectivity_score(weight_mask_list_last, debug_flag=debug_flag)
        mean_last_connectivity_score /= num_seeds
        strategies_connectivity_dict[strategy].append(mean_last_connectivity_score)  # after last expansion the network is fully connected
        print("------------------------- Strategy ", strategy, " done ----------------------------")

    torch.save(strategies_connectivity_dict,
               "../results/" + f"Connectivity_scores_{dataset_name}_initial_network_{initial_network_id}_num_seeds_{num_seeds}_expansions_{num_expansions}_eps_{eps}_lr_{lr}_grad_clip_{grad_clip}.pt")

