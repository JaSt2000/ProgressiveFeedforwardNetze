import numpy as np
import torch
import include.Model_and_Methods as mm


def random_neurons_expand(device, model, num_total_expansions, num_expansions):
    num_neurons_total = model.neurons_total()

    flag_actual_expansion = False
    idx_to_choose_from = mm.get_available_neurons_global(model.weight_mask_list, model.bias_mask_list, num_neurons_total)
    idx_to_choose_from = torch.Tensor(idx_to_choose_from).to(torch.int32).to(device)
    num_neurons_to_be_added = int(model.total_neurons_to_be_added.sum().item() / num_total_expansions)

    if torch.numel(idx_to_choose_from) > 0:  # there are still neurons to be added
        flag_actual_expansion = True

        # last expansion
        if num_expansions == num_total_expansions-1:
            mm.init_all_masked_params(device, model)
            return flag_actual_expansion

        if torch.numel(idx_to_choose_from) >= num_neurons_to_be_added:
            # shuffle indices
            shuffled_idx_to_choose_from = idx_to_choose_from[torch.randperm(idx_to_choose_from.size(dim=0))]
            chosen_idx = shuffled_idx_to_choose_from[:num_neurons_to_be_added]
        else:
            # add all remaining neurons
            chosen_idx = idx_to_choose_from

        for new_neuron_index in chosen_idx:
            new_neuron_layer_index, new_neuron_row_index = mm.get_neuron_model_indices_from_global_array(new_neuron_index, model.weight_mask_list)
            mm.initNeuron(device, model, new_neuron_layer_index, new_neuron_row_index)
    return flag_actual_expansion
