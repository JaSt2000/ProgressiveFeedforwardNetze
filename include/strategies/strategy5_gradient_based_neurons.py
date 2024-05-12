import numpy as np
import torch
from torch.utils.data import DataLoader
import include.Model_and_Methods as mm
import torch.nn as nn
from copy import deepcopy


def compute_neuron_gradients(device, model, training_data):  # splitting vectors are returned by the function
    batch_size = 100
    criterion = nn.CrossEntropyLoss()
    data_loader = torch.utils.data.DataLoader(dataset=training_data,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              generator=torch.Generator(device='cuda'))
    gradients = [torch.zeros(size=(elem.size(dim=0),)) for elem in
                 model.weight_mask_list]
    # Loop over each batch from the training set
    for batch_idx, (data, target) in enumerate(data_loader):
        # Copy data to GPU if needed
        data = data.to(device)
        target = target.to(device)

        # Pass data through the network
        layer_outputs = model(data)

        # Compute loss
        loss = criterion(layer_outputs[-1], target)

        # zero out gradients wrt layer_outputs if needed, retain layer_outputs as they are non_leaf nodes (created by an operation tracked by autograd)
        for layer_index in range(len(layer_outputs)):
            layer_outputs[layer_index].retain_grad()
            if layer_outputs[layer_index].grad is not None:
                layer_outputs[layer_index].grad.zero_()

        # Gradient computation
        loss.backward()

        for layer_index in range(len(model.layers)):
            for row_index in range(model.weight_mask_list[layer_index].size(dim=0)):
                gradients[layer_index][row_index] += layer_outputs[layer_index].grad[:, row_index].mean()
    for layer_index in range(len(model.layers)):
        for row_index in range(model.weight_mask_list[layer_index].size(dim=0)):
            gradients[layer_index][row_index] /= len(data_loader)

    return gradients


def gradient_based_neurons_expand(device, model,
                             num_total_expansions,
                             training_data, flag_use_gradient_values, num_expansions):
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

    gradients = compute_neuron_gradients(device, model, training_data)

    #compute gradient norms (absolute values)
    gradient_norms = [torch.zeros(size=(elem.size(dim=0),)) for elem in
                 model.weight_mask_list]
    for layer_index in range(len(model.layers)):
        for row_index in range(model.weight_mask_list[layer_index].size(dim=0)):
            gradient_norms[layer_index][row_index] = torch.abs(gradients[layer_index][row_index])

    number_of_neurons_in_layer = [model.weight_mask_list[k].size(dim=0) - len(
        mm.get_available_neurons_in_layer(k, model.weight_mask_list, model.bias_mask_list)) for k in
                                  range(len(model.weight_mask_list))]
    gradient_norms_mean_over_neurons = [torch.sum(gradient_norms[i])/np.max([number_of_neurons_in_layer[i], 1]) for i in range(len(gradient_norms))]

    if flag_use_gradient_values:
        layer_statistics = torch.tensor(gradient_norms_mean_over_neurons)
    else:  # only use rank in sorted list
        layer_statistics = torch.tensor(gradient_norms_mean_over_neurons)
        values, indices = torch.sort(layer_statistics, descending=False)
        for i in range(len(indices)):
            layer_statistics[indices[i]] = i+1

    relative_units_to_be_added = layer_statistics / torch.sum(layer_statistics)
    # relative_units_to_be_added[relative_units_to_be_added == 0] = 1 / num_neurons_to_be_added  # does not sum up to 1 anymore but does not matter

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

    print("neurons added through strategy (gradient_based_neurons): ", num_neurons_added/num_neurons_to_be_added)
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
