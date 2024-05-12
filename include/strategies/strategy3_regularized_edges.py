import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import include.Model_and_Methods as mm
from copy import deepcopy


def regularize_edges(device, model, layer_index, edge_indices, abs_list, training_data):
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

    for i in edge_indices:
        row_index, column_index, is_weight = mm.get_edge_model_indices_from_layer_array(layer_index, i,
                                                                                        model.weight_mask_list,
                                                                                        model.bias_mask_list)
        # initialize considered edge
        init_boundary = np.sqrt(
            6 / (model.weight_mask_list[layer_index].size(dim=0) + model.weight_mask_list[layer_index].size(dim=1)))
        if is_weight:
            reg_weight_mask_list[layer_index][row_index, column_index] = 1
            with torch.no_grad():
                torch.nn.init.uniform_(model.layers[layer_index].weight[row_index, column_index],
                                       a=-init_boundary,
                                       b=init_boundary)

        else:  # bias param
            reg_bias_mask_list[layer_index][row_index] = 1
            with torch.no_grad():
                torch.nn.init.uniform_(model.layers[layer_index].bias[row_index],
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

            # Add L1 regularization (group sparsity reg not needed for edge regularization)
            l1_reg = torch.linalg.vector_norm(model.layers[layer_index].weight,
                                              ord=1) + torch.linalg.vector_norm(
                model.layers[layer_index].bias, ord=1)

            loss += 0.1 * l1_reg

            # Backpropagate - gradient computation
            loss.backward()

            # zero out gradients of not considered edges
            for l in range(len(model.layers)):
                model.layers[l].weight.grad *= reg_weight_mask_list[l]
                model.layers[l].bias.grad *= reg_bias_mask_list[l]

            # Update weights
            optimizer.step()

    # save absolute weight values and zero out the regularized weights
    with torch.no_grad():
        for i in edge_indices:
            row_index, column_index, is_weight = mm.get_edge_model_indices_from_layer_array(layer_index, i,
                                                                                            model.weight_mask_list,
                                                                                            model.bias_mask_list)
            array_idx = mm.get_global_index_edges_array(layer_index, row_index, column_index, is_weight,
                                                        model.weight_mask_list, model.bias_mask_list)
            if is_weight:
                abs_list[array_idx] = torch.abs(model.layers[layer_index].weight[row_index, column_index])
            else:
                abs_list[array_idx] = torch.abs(model.layers[layer_index].bias[row_index])

        for i in range(len(model.layers)):
            model.layers[i].weight *= model.weight_mask_list[i]
            model.layers[i].bias *= model.bias_mask_list[i]


def regularized_edges_expand(device, model, num_total_expansions, training_data,
                             num_expansions):  # add k edges to each layer and then regularize them
    num_params_total = model.params_total()

    result_abs_list_from_regularization = (-1) * torch.ones(size=(num_params_total,)).to(torch.float32).to(
        device)  # -1 for existing neurons or neurons that are not considered
    num_edges_to_be_added = np.max([int(model.total_params_to_be_added.sum().item() / num_total_expansions), 1])
    flag_actual_expansion = False

    list_all_available_edges = mm.get_available_edges_global(model.weight_mask_list, model.bias_mask_list,
                                                             num_params_total)

    if len(list_all_available_edges) <= num_edges_to_be_added:
        if len(list_all_available_edges) > 0:
            flag_actual_expansion = True

            chosen_new_edges = list_all_available_edges
            for new_edge_index in chosen_new_edges:
                layer_idx, row_idx, column_idx, is_weight = mm.get_edge_model_indices_from_global_array(new_edge_index,
                                                                                                        model.weight_mask_list,
                                                                                                        model.bias_mask_list)
                mm.initWeight(device, model, layer_idx, row_idx, column_idx,
                              is_weight)
    else:
        flag_actual_expansion = True

        # last expansion
        if num_expansions == num_total_expansions - 1:
            mm.init_all_masked_params(device, model)
            return flag_actual_expansion

        for layer_index in range(len(model.weight_mask_list)):
            num_const_edges_this_layer = 2 * np.max(
                [int(model.total_params_to_be_added[layer_index] / num_total_expansions), 1])
            idx_to_choose_from = mm.get_available_edges_in_layer(layer_index, model.weight_mask_list,
                                                                 model.bias_mask_list)

            if len(idx_to_choose_from) > 0:
                chosen_edge_idx = idx_to_choose_from[:num_const_edges_this_layer]
                regularize_edges(device, model, layer_index, chosen_edge_idx, result_abs_list_from_regularization,
                                 training_data)
                print("regularization layer ", layer_index, " done")

        values, indices = torch.sort(result_abs_list_from_regularization, descending=True)
        chosen_new_edges = indices[:num_edges_to_be_added]
        for new_edge_index in chosen_new_edges:
            # avoid re-initializing already existing neurons (or neurons that are not considered)
            if result_abs_list_from_regularization[new_edge_index] >= 0:
                layer_idx, row_idx, column_idx, is_weight = mm.get_edge_model_indices_from_global_array(new_edge_index,
                                                                                                        model.weight_mask_list,
                                                                                                        model.bias_mask_list)
                mm.initWeight(device, model, layer_idx, row_idx, column_idx,
                              is_weight)
    return flag_actual_expansion
