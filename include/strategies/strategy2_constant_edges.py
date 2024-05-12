import numpy as np
import torch
import include.Model_and_Methods as mm


def constant_edges_expand(device, model, num_total_expansions, num_expansions):
    flag_actual_expansion = False
    for layer_index in range(len(model.weight_mask_list)):
        num_edges_to_be_added = int(model.total_params_to_be_added[layer_index] / num_total_expansions)
        idx_to_choose_from = mm.get_available_edges_in_layer(layer_index, model.weight_mask_list,
                                                             model.bias_mask_list)  # already in ascending order
        idx_to_choose_from = torch.Tensor(idx_to_choose_from).to(torch.int32).to(device)

        if torch.numel(idx_to_choose_from) > 0:  # there are still edges to be added
            flag_actual_expansion = True

            # last expansion
            if num_expansions == num_total_expansions - 1:
                mm.init_all_masked_params(device, model)
                return flag_actual_expansion

            chosen_edge_idx = idx_to_choose_from[:num_edges_to_be_added]
            for elem in chosen_edge_idx:
                row_index, column_index, is_weight = mm.get_edge_model_indices_from_layer_array(layer_index, elem,
                                                                                                model.weight_mask_list,
                                                                                                model.bias_mask_list)
                mm.initWeight(device, model, layer_index, row_index, column_index, is_weight)
    return flag_actual_expansion
