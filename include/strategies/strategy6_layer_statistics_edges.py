import numpy as np
import torch
from torch.utils.data import DataLoader
import include.Model_and_Methods as mm
from copy import deepcopy


def compute_layer_statistics(device, model, training_data):  # splitting vectors are returned by the function
    batch_size = 100
    data_loader = torch.utils.data.DataLoader(dataset=training_data,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              generator=torch.Generator(device='cuda'))
    variances = [torch.zeros(size=(elem.size(dim=0),)) for elem in
                 model.weight_mask_list]  # compute second moment first here, then use mean to compute variance
    means = [torch.zeros(size=(elem.size(dim=0),)) for elem in model.weight_mask_list]
    with torch.no_grad():
        # Loop over each batch from the training set
        for batch_idx, (data, target) in enumerate(data_loader):
            # Copy data to GPU if needed
            data = data.to(device)
            target = target.to(device)

            # Pass data through the network
            layer_outputs = model(data)
            layer_outputs_mean = [torch.sum(i, dim=0) / batch_size for i in layer_outputs]
            layer_outputs_squared_mean = [torch.sum(i ** 2, dim=0) / batch_size for i in layer_outputs]
            for l in range(len(layer_outputs)):
                means[l] += layer_outputs_mean[l]
                variances[l] += layer_outputs_squared_mean[l]
        for l in range(len(model.weight_mask_list)):
            means[l] /= len(data_loader)
            variances[l] /= len(
                data_loader)  # this coresponds to the second moment, now compute variance with mean

        for l in range(len(model.weight_mask_list)):
            variances[l] -= means[l] ** 2  # this is the variance
    return means, variances


def layer_stat_edges_expand(device, model,
                             num_total_expansions,
                             training_data, flag_use_variance_values, num_expansions):
    flag_actual_expansion = False

    num_neurons_total = model.neurons_total()

    num_params_total = model.params_total()

    list_all_available_edges = mm.get_available_edges_global(model.weight_mask_list, model.bias_mask_list, num_params_total)

    if len(list_all_available_edges) == 0:
        return flag_actual_expansion

    if num_expansions == num_total_expansions - 1:  # last expansion
        flag_actual_expansion = True
        mm.init_all_masked_params(device, model)
        return flag_actual_expansion

    num_params_to_be_added = np.max([int(model.total_params_to_be_added.sum().item() / num_total_expansions), 1])

    _, variances = compute_layer_statistics(device, model, training_data)
    variances_each_neuron = -torch.ones(size=(num_neurons_total, ))
    for i in range(num_neurons_total):
        layer_index, row_index = mm.get_neuron_model_indices_from_global_array(i, model.weight_mask_list)
        variances_each_neuron[i] = variances[layer_index][row_index]


    if flag_use_variance_values:
        neuron_statistics = 1/variances_each_neuron
        neuron_statistics = torch.nan_to_num(neuron_statistics, nan=0, posinf=0, neginf=0)
        max_val = torch.max(neuron_statistics)
        neuron_statistics[neuron_statistics == 0] = max_val + 1
    else:  # only use rank in sorted list
        neuron_statistics = variances_each_neuron
        values, indices = torch.sort(neuron_statistics, descending=True)
        for i in range(len(indices)):
            neuron_statistics[indices[i]] = i+1

    relative_units_to_be_added = neuron_statistics / torch.sum(neuron_statistics)  # to each neuron

    values, indices = torch.sort(neuron_statistics, descending=True)

    # check in which layers new neurons can be added
    num_params_added = 0
    for neuron_index in indices:
        layer_index, row_index = mm.get_neuron_model_indices_from_global_array(neuron_index, model.weight_mask_list)
        list_available_edges_in_neuron, bias_available = mm.get_available_edges_in_neuron(layer_index, row_index, model.weight_mask_list, model.bias_mask_list)
        num_params_to_be_added_this_neuron = np.min([len(list_available_edges_in_neuron) + int(bias_available), int(relative_units_to_be_added[neuron_index] * num_params_to_be_added)])
        bias_index = len(list_available_edges_in_neuron)
        if bias_available:
            list_available_edges_in_neuron.append(bias_index)  # represents bias value
        np.random.shuffle(list_available_edges_in_neuron)
        chosen_edges = list_available_edges_in_neuron[
                       :num_params_to_be_added_this_neuron]
        for column_index in chosen_edges:
            if column_index == bias_index:
                mm.initWeight(device, model, layer_index, row_index, -1, False)
            else:
                mm.initWeight(device, model, layer_index, row_index, column_index, True)
            flag_actual_expansion = True
            num_params_added += 1
            if num_params_added == num_params_to_be_added:
                break
        else:
            continue  # only executed when inner loop did not break
        break

    print("edges added through strategy (layer_stat_edges): ", num_params_added/num_params_to_be_added)
    if num_params_added < num_params_to_be_added:  # this avoids that very little amounts of edges are added during an expansion due to a neuron thats almost completely added with a very low variance
        for neuron_index in indices:
            layer_index, row_index = mm.get_neuron_model_indices_from_global_array(neuron_index, model.weight_mask_list)
            list_available_edges_in_neuron, bias_available = mm.get_available_edges_in_neuron(layer_index, row_index,
                                                                                              model.weight_mask_list,
                                                                                              model.bias_mask_list)
            chosen_edges = list_available_edges_in_neuron
            for column_index in chosen_edges:
                mm.initWeight(device, model, layer_index, row_index, column_index, True)
                flag_actual_expansion = True
                num_params_added += 1
                if num_params_added == num_params_to_be_added:
                    break
            if bias_available and num_params_added < num_params_to_be_added:
                mm.initWeight(device, model, layer_index, row_index, -1, False)
                flag_actual_expansion = True
                num_params_added += 1
            if num_params_added == num_params_to_be_added:
                break
    return flag_actual_expansion
