import numpy as np
import torch
import include.Model_and_Methods as mm


def get_abs_edges_array(model_initial, weight_mask_list, bias_mask_list, abs_list):
    for i in range(len(abs_list)):
        layer_index, row_index, column_index, is_weight = mm.get_edge_model_indices_from_global_array(i,
                                                                                                      weight_mask_list,
                                                                                                      bias_mask_list)
        if is_weight:
            abs_list[i] = torch.abs(model_initial.layers[layer_index].weight[row_index, column_index])
        else:
            abs_list[i] = torch.abs(model_initial.layers[layer_index].bias[row_index])

    return


def get_abs_neurons_array(model_initial, weight_mask_list,
                          abs_list):  # take the mean over the absolute values of all parameters of a neurons
    for i in range(len(abs_list)):
        layer_index, row_index = mm.get_neuron_model_indices_from_global_array(i,
                                                                               weight_mask_list)
        abs_list[i] = (torch.sum(torch.abs(model_initial.layers[layer_index].weight[row_index, :])) + torch.abs(
            model_initial.layers[layer_index].bias[row_index])) / (weight_mask_list[layer_index].size(dim=1) + 1)

    return


def initial_neurons_expand(device, model, num_total_expansions, model_initial, num_expansions):
    num_neurons_total = model.neurons_total()

    result_abs_list_from_initial_model = (-1) * torch.ones(size=(num_neurons_total,)).to(torch.float32).to(
        device)
    get_abs_neurons_array(model_initial, model.weight_mask_list,
                          result_abs_list_from_initial_model)  # fill the absolute values array
    flag_actual_expansion = False
    idx_to_choose_from = mm.get_available_neurons_global(model.weight_mask_list, model.bias_mask_list,
                                                         num_neurons_total)
    idx_to_choose_from = torch.Tensor(idx_to_choose_from).to(torch.int32).to(device)
    abs_values_available_neurons = result_abs_list_from_initial_model[idx_to_choose_from]
    num_neurons_to_be_added = int(model.total_neurons_to_be_added.sum().item() / num_total_expansions)

    values, indices = torch.sort(abs_values_available_neurons, descending=True)

    chosen_indices = indices[:num_neurons_to_be_added]
    if torch.numel(
            indices) > 0:  # there are still neurons to be added
        flag_actual_expansion = True

        # last expansion
        if num_expansions == num_total_expansions - 1:
            mm.init_all_masked_params(device, model)
            return flag_actual_expansion

        for elem in chosen_indices:
            global_index = idx_to_choose_from[elem]  # get global edge index
            layer_index, row_index = mm.get_neuron_model_indices_from_global_array(global_index,
                                                                                   model.weight_mask_list)
            mm.initNeuron(device, model, layer_index, row_index)
    return flag_actual_expansion
