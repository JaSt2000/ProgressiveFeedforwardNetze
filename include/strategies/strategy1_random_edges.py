import numpy as np
import torch
import include.Model_and_Methods as mm


def random_edges_expand(device, model, num_total_expansions, num_expansions):
    flag_actual_expansion = False
    num_params_total = model.params_total()
    idx_to_choose_from = mm.get_available_edges_global(model.weight_mask_list, model.bias_mask_list, num_params_total)
    idx_to_choose_from = torch.Tensor(idx_to_choose_from).to(torch.int32).to(device)
    num_edges_to_be_added = int(model.total_params_to_be_added.sum().item() / num_total_expansions)
    # print("test print number: ", num_edges_to_be_added)

    if torch.numel(idx_to_choose_from) > 0:  # there are still edges to be added
        flag_actual_expansion = True

        # last expansion
        if num_expansions == num_total_expansions - 1:
            mm.init_all_masked_params(device, model)
            return flag_actual_expansion

        if torch.numel(idx_to_choose_from) >= num_edges_to_be_added:
            # shuffle indices
            shuffled_idx_to_choose_from = idx_to_choose_from[torch.randperm(idx_to_choose_from.size(dim=0))]
            chosen_idx = shuffled_idx_to_choose_from[:num_edges_to_be_added]
        else:
            # add all remaining edges
            chosen_idx = idx_to_choose_from

        for elem in chosen_idx:
            layer_index, row_index, column_index, is_weight = mm.get_edge_model_indices_from_global_array(elem, model.weight_mask_list, model.bias_mask_list)
            mm.initWeight(device, model, layer_index, row_index, column_index, is_weight)
    return flag_actual_expansion
