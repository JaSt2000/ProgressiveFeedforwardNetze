import numpy as np
import torch
import include.Model_and_Methods as mm


def constant_neurons_expand(device, model, num_total_expansions,
                            num_expansions):  # add k neurons to each layer (randomly) and initialize them randomly
    flag_actual_expansion = False
    for layer_index in range(len(model.weight_mask_list)):
        num_neurons_to_be_added = np.max([int(model.total_neurons_to_be_added[layer_index] / num_total_expansions), 1])
        idx_to_choose_from = mm.get_available_neurons_in_layer(layer_index, model.weight_mask_list,
                                                               model.bias_mask_list)  # in ascending order
        idx_to_choose_from = torch.Tensor(idx_to_choose_from).to(torch.int32).to(device)

        if torch.numel(idx_to_choose_from) > 0:  # there are still neurons to be added
            flag_actual_expansion = True

            # last expansion
            if num_expansions == num_total_expansions - 1:
                mm.init_all_masked_params(device, model)
                return flag_actual_expansion

            chosen_neuron_idx = idx_to_choose_from[:num_neurons_to_be_added]
            for elem in chosen_neuron_idx:
                mm.initNeuron(device, model, layer_index, elem)
    return flag_actual_expansion
