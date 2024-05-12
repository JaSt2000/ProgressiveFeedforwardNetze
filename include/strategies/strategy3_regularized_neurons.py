import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import include.Model_and_Methods as mm
from copy import deepcopy


# def get_num_neurons_considered_total(weight_mask_list, bias_mask_list, num_total_expansions, num_additional_neurons_per_layer):
#     num_neurons_considered = 0
#     for l in range(len(weight_mask_list)):
#         num_const_neurons_layer_l = 2 * np.max(
#             [int(weight_mask_list[l].size(dim=0) / num_total_expansions), 1]) + num_additional_neurons_per_layer
#         num_available_neurons_layer_l = len(mm.get_available_neurons_in_layer(l, weight_mask_list, bias_mask_list))
#         num_neurons_considered += np.min([num_const_neurons_layer_l, num_available_neurons_layer_l])
#     return num_neurons_considered
def regularize_neurons(device, model, layer_index, neuron_indices, abs_list, training_data):
    batch_size = 100
    train_loader = torch.utils.data.DataLoader(dataset=training_data,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               generator=torch.Generator(device='cuda'))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.2)
    criterion = nn.CrossEntropyLoss()  # add regularization later

    # define via mask what parameters are optimized
    reg_weight_mask_list = []
    reg_bias_mask_list = []
    for l in range(len(model.weight_mask_list)):
        reg_weight_mask_list.append(torch.zeros_like(model.weight_mask_list[l]))
        reg_bias_mask_list.append(torch.zeros_like(model.bias_mask_list[l]))

    for i in neuron_indices:
        reg_weight_mask_list[layer_index][i, :] = torch.ones(
            size=(model.weight_mask_list[layer_index].size(dim=1),), device=device, dtype=torch.int32)
        reg_bias_mask_list[layer_index][i] = 1
        init_boundary = np.sqrt(
            6 / (model.weight_mask_list[layer_index].size(dim=0) + model.weight_mask_list[
                layer_index].size(
                dim=1)))
        with torch.no_grad():
            torch.nn.init.uniform_(
                model.layers[layer_index].weight[i, :],
                a=-init_boundary,
                b=init_boundary)
            torch.nn.init.uniform_(model.layers[layer_index].bias[i],
                                   a=-init_boundary,
                                   b=init_boundary)

    # Set model to training mode
    model.train()

    for epoch in range(2):  # use 2 epochs for regularization

        # Loop over each batch from the training set
        for batch_idx, (data, target) in enumerate(train_loader):
            # Copy data to GPU if needed
            data = data.to(device)
            target = target.to(device)

            # Zero gradient buffers
            optimizer.zero_grad()

            # Pass data through the network
            output = model(data)[-1]

            # Calculate loss
            loss = criterion(output, target)

            # Add L1 regularization and group sparsity regularization for the considered layer
            l1_reg = torch.linalg.vector_norm(model.layers[layer_index].weight,
                                              ord=1) + torch.linalg.vector_norm(
                model.layers[layer_index].bias, ord=1)
            group_reg = 0
            for i in neuron_indices:
                group_reg += torch.sqrt(
                    torch.linalg.vector_norm(model.layers[layer_index].weight[i, :],
                                             ord=2) ** 2 + torch.linalg.vector_norm(
                        model.layers[layer_index].bias[i], ord=2) ** 2)

            loss += 0.1 * l1_reg + 0.01 * group_reg

            # Backpropagate - gradient computation
            loss.backward()

            # zero out gradients of not considered edges
            for i in range(len(model.layers)):
                    model.layers[i].weight.grad *= reg_weight_mask_list[i]
                    model.layers[i].bias.grad *= reg_bias_mask_list[i]

            # Update weights
            optimizer.step()

    # save absolute weight values and zero out the regularized weights
    with torch.no_grad():
        for i in neuron_indices:
            array_idx = mm.get_global_index_neuron_array(layer_index, i, model.weight_mask_list)
            abs_list[array_idx] = (torch.sum(torch.abs(model.layers[layer_index].weight[i, :])) + torch.abs(
                model.layers[layer_index].bias[i]))/(model.weight_mask_list[layer_index].size(dim=1) + 1)

        for i in range(len(model.layers)):
            model.layers[i].weight *= model.weight_mask_list[i]
            model.layers[i].bias *= model.bias_mask_list[i]


def regularized_neurons_expand(device, model, num_total_expansions, training_data,
                               num_expansions):  # add k neurons to each layer and then regularize them
    num_neurons_total = model.neurons_total()

    result_abs_list_from_regularization = (-1) * torch.ones(size=(num_neurons_total,)).to(torch.float32).to(
        device)  # -1 for existing neurons or neurons that are not considered
    num_neurons_to_be_added = np.max([int(model.total_neurons_to_be_added.sum().item() / num_total_expansions), 1])
    # print("number of neurons total: ",  num_neurons_total)
    flag_actual_expansion = False

    list_all_available_neurons = mm.get_available_neurons_global(model.weight_mask_list, model.bias_mask_list,
                                                                 num_neurons_total)
    # print("number of remaining free neurons: ", len(list_all_available_neurons))

    if len(list_all_available_neurons) <= num_neurons_to_be_added:
        if len(list_all_available_neurons) > 0:
            flag_actual_expansion = True
            chosen_new_neurons = list_all_available_neurons
            for new_neuron_index in chosen_new_neurons:
                new_neuron_layer_index, new_neuron_row_index = mm.get_neuron_model_indices_from_global_array(
                    new_neuron_index, model.weight_mask_list)
                mm.initNeuron(device, model, new_neuron_layer_index,
                              new_neuron_row_index)

    else:
        flag_actual_expansion = True

        # last expansion
        if num_expansions == num_total_expansions - 1:
            mm.init_all_masked_params(device, model)
            return flag_actual_expansion

        # in case there need to be more neurons considered (exactly num_neurons_to_be_added in total)----------

        # num_additional_neurons_per_layer = 0
        # num_neurons_considered = get_num_neurons_considered_total(model.weight_mask_list, model.bias_mask_list, num_total_expansions, num_additional_neurons_per_layer)
        # while num_neurons_considered < num_neurons_to_be_added:
        #     num_additional_neurons_per_layer += 1
        #     num_neurons_considered = get_num_neurons_considered_total(model.weight_mask_list, model.bias_mask_list, num_total_expansions, num_additional_neurons_per_layer)
        # -----------------------------------------------------------------------------------------------------
        for layer_index in range(len(model.weight_mask_list)):
            num_const_neurons_this_layer = 2 * np.max(
                [int(model.total_neurons_to_be_added[layer_index] / num_total_expansions),
                 1])  # + num_additional_neurons_per_layer
            idx_to_choose_from = mm.get_available_neurons_in_layer(layer_index, model.weight_mask_list,
                                                                   model.bias_mask_list)
            if len(idx_to_choose_from) > 0:
                chosen_neuron_idx = idx_to_choose_from[:num_const_neurons_this_layer]
                regularize_neurons(device, model, layer_index, chosen_neuron_idx, result_abs_list_from_regularization,
                                   training_data)
                print("regularization layer ", layer_index, " done")

        values, indices = torch.sort(result_abs_list_from_regularization, descending=True)
        chosen_new_neurons = indices[:num_neurons_to_be_added]
        for new_neuron_index in chosen_new_neurons:
            # avoid re-initializing already existing neurons (or neurons that are not considered)
            if result_abs_list_from_regularization[new_neuron_index] >= 0:
                new_neuron_layer_index, new_neuron_row_index = mm.get_neuron_model_indices_from_global_array(
                    new_neuron_index, model.weight_mask_list)
                mm.initNeuron(device, model, new_neuron_layer_index,
                              new_neuron_row_index)
    return flag_actual_expansion
