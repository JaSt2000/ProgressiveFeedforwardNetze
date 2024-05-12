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


def layer_stat_neurons_expand(device, model,
                             num_total_expansions,
                             training_data, flag_use_variance_values, num_expansions):
    flag_actual_expansion = False

    num_neurons_total = model.neurons_total()

    num_params_total = model.params_total()

    list_all_available_neurons = mm.get_available_neurons_global(model.weight_mask_list, model.bias_mask_list, num_neurons_total)

    if len(list_all_available_neurons) == 0:
        return flag_actual_expansion

    if num_expansions == num_total_expansions - 1:  # last expansion
        flag_actual_expansion = True
        mm.init_all_masked_params(device, model)
        return flag_actual_expansion

    num_neurons_to_be_added = np.max([int(model.total_neurons_to_be_added.sum().item() / num_total_expansions), 1])

    _, variances = compute_layer_statistics(device, model, training_data)
    number_of_neurons_in_layer = [model.weight_mask_list[k].size(dim=0) - len(mm.get_available_neurons_in_layer(k, model.weight_mask_list, model.bias_mask_list)) for k in range(len(model.weight_mask_list))]
    variance_mean_over_neurons = [torch.sum(variances[i])/np.max([number_of_neurons_in_layer[i], 1]) for i in range(len(variances))]

    if flag_use_variance_values:
        layer_statistics = torch.tensor(variance_mean_over_neurons)
        layer_statistics = 1/layer_statistics
        layer_statistics = torch.nan_to_num(layer_statistics, nan=0, posinf=0, neginf=0)
        max_val = torch.max(layer_statistics)
        layer_statistics[layer_statistics == 0] = max_val + 1
    else:  # only use rank in sorted list
        layer_statistics = torch.tensor(variance_mean_over_neurons)
        values, indices = torch.sort(layer_statistics, descending=True)
        for i in range(len(indices)):
            layer_statistics[indices[i]] = i+1

    relative_units_to_be_added = layer_statistics / torch.sum(layer_statistics)

    values, indices = torch.sort(layer_statistics, descending=True)

    # check in which layers new neurons can be added
    num_neurons_added = 0
    for layer_index in indices:
        list_available_neurons_in_layer = mm.get_available_neurons_in_layer(layer_index, model.weight_mask_list,
                                                                        model.bias_mask_list)
        num_neurons_to_be_added_this_layer = np.min([len(list_available_neurons_in_layer), int(relative_units_to_be_added[layer_index] * num_neurons_to_be_added)])

        chosen_neurons = list_available_neurons_in_layer[
                       :num_neurons_to_be_added_this_layer]
        for row_index in chosen_neurons:
            mm.initNeuron(device, model, layer_index, row_index)
            flag_actual_expansion = True
            num_neurons_added += 1
            if num_neurons_added == num_neurons_to_be_added:
                break
        else:
            continue  # only executed when inner loop did not break
        break

    print("neurons added through strategy (layer_stat_neurons): ", num_neurons_added/num_neurons_to_be_added)
    if num_neurons_added < num_neurons_to_be_added:  # this avoids that very little amounts of neurons are added during an expansion due to a layer thats almost completely added with a very low variance
        for layer_index in indices:
            list_available_neurons_in_layer = mm.get_available_neurons_in_layer(layer_index, model.weight_mask_list,
                                                                                model.bias_mask_list)
            chosen_neurons = list_available_neurons_in_layer
            for row_index in chosen_neurons:
                mm.initNeuron(device, model, layer_index, row_index)
                flag_actual_expansion = True
                num_neurons_added += 1
                if num_neurons_added == num_neurons_to_be_added:
                    break
            else:
                continue  # only executed when inner loop did not break
            break
    return flag_actual_expansion
