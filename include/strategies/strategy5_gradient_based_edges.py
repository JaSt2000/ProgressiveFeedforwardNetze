import numpy as np
import torch
from torch.utils.data import DataLoader
import include.Model_and_Methods as mm
from copy import deepcopy
import torch.nn as nn


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


def gradient_based_edges_expand(device, model,
                             num_total_expansions,
                             training_data, flag_use_gradient_values, num_expansions):
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

    gradients = compute_neuron_gradients(device, model, training_data)

    # compute gradient norms (absolute values)
    gradient_norms = [torch.zeros(size=(elem.size(dim=0),)) for elem in
                      model.weight_mask_list]
    for layer_index in range(len(model.layers)):
        for row_index in range(model.weight_mask_list[layer_index].size(dim=0)):
            gradient_norms[layer_index][row_index] = torch.abs(gradients[layer_index][row_index])


    gradient_norm_each_neuron = -torch.ones(size=(num_neurons_total, ))
    for i in range(num_neurons_total):
        layer_index, row_index = mm.get_neuron_model_indices_from_global_array(i, model.weight_mask_list)
        gradient_norm_each_neuron[i] = gradient_norms[layer_index][row_index]

    # print(gradient_norm_each_neuron)

    if flag_use_gradient_values:
        neuron_statistics = gradient_norm_each_neuron
    else:  # only use rank in sorted list
        neuron_statistics = gradient_norm_each_neuron
        values, indices = torch.sort(neuron_statistics, descending=False)
        for i in range(len(indices)):
            neuron_statistics[indices[i]] = i+1

    relative_units_to_be_added = neuron_statistics / torch.sum(neuron_statistics)  # to each neuron
    # relative_units_to_be_added[relative_units_to_be_added == 0] = 1/num_params_to_be_added    # does not sum up to 1 anymore but does not matter
    # print((relative_units_to_be_added * num_params_total).to(torch.int32))
    # print(relative_units_to_be_added[0])
    values, indices = torch.sort(relative_units_to_be_added, descending=True)
    # check in which layers new neurons can be added
    num_params_added = 0
    for neuron_index in indices:
        layer_index, row_index = mm.get_neuron_model_indices_from_global_array(neuron_index, model.weight_mask_list)
        list_available_edges_in_neuron, bias_available = mm.get_available_edges_in_neuron(layer_index, row_index, model.weight_mask_list, model.bias_mask_list)
        num_params_to_be_added_this_neuron = np.min([len(list_available_edges_in_neuron) + int(bias_available), int(relative_units_to_be_added[neuron_index] * num_params_to_be_added)])
        # print(f"{relative_units_to_be_added[neuron_index]} * {num_params_to_be_added} = {relative_units_to_be_added[neuron_index] * num_params_to_be_added} = {int(relative_units_to_be_added[neuron_index] * num_params_to_be_added)} (integer cast)")
        # print("num_params_to_be_added_this_neuron: ", len(list_available_edges_in_neuron) + int(bias_available))
        bias_index = len(list_available_edges_in_neuron)
        if bias_available:
            list_available_edges_in_neuron.append(bias_index)  #represents bias value
        np.random.shuffle(list_available_edges_in_neuron)
        chosen_edges = list_available_edges_in_neuron[
                       :num_params_to_be_added_this_neuron]
        for column_index in chosen_edges:
            if column_index == bias_index:  # list has gotten 1 element
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

    print("edges added through strategy (gradient_based_edges): ", num_params_added/num_params_to_be_added)
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
    print("edges added total (gradient_based_edges): ", num_params_added / num_params_to_be_added)
    return flag_actual_expansion
