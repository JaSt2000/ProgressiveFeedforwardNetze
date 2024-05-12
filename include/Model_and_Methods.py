import struct
from array import array
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


# MNIST Data Loader Class
#
class MnistDataloader(object):
    def __init__(self, training_images_filepath, training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    def read_images_labels(self, images_filepath, labels_filepath):
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())

        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img

        return images, labels

    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train), (x_test, y_test)


# torch dataset class for mnist
class PytorchDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self.x = x
        self.y = y

    def __getitem__(self, idx):
        return self.x[idx, :], self.y[idx]

    def __len__(self):
        return len(self.y)


# baseline model definition
class ProgressiveMLP(nn.Module):
    def __init__(self, input_size=28 * 28, n_layers=5, n_neurons=30, activation=nn.ReLU(), num_classes=10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.ModuleList()
        self.acts = nn.ModuleList()

        self.layers.append(nn.Linear(input_size, n_neurons))
        self.acts.append(activation)
        for i in range(n_layers - 2):  # flexibler als nn.Sequential
            self.layers.append(nn.Linear(n_neurons, n_neurons))
            self.acts.append(activation)
            # self.add_module(f"layer{i}", self.layers[-1]) #only necessary when using python list instead of ModuleList
            # self.add_module(f"act{i}", self.acts[-1])
        self.layers.append(nn.Linear(n_neurons, num_classes))

        self.weight_mask_list = [torch.ones_like(layer.weight) for layer in self.layers] # initial state: no masks/ entire network
        self.bias_mask_list = [torch.ones_like(layer.bias) for layer in self.layers]

        self.total_params_to_be_added = torch.zeros(size=(n_layers, ))
        self.total_neurons_to_be_added = torch.zeros(size=(n_layers, ))

    def neurons_total(self):
        num_neurons_total = 0
        for mask in self.weight_mask_list:
            num_neurons_total += mask.size(dim=0)
        return num_neurons_total

    def params_total(self):
        num_params_total = 0
        for l in range(len(self.weight_mask_list)):
            num_params_total += torch.numel(self.weight_mask_list[l]) + torch.numel(self.bias_mask_list[l])
        return num_params_total

    def param_count(self):
        num_params_current = 0
        for l in range(len(self.weight_mask_list)):
            num_params_current += torch.count_nonzero(self.weight_mask_list[l]) + torch.count_nonzero(
                self.bias_mask_list[l])
        return num_params_current

    def neuron_count(self):
        num_neurons_current = 0
        for layer_index in range(len(self.weight_mask_list)):
            for row_index in range(self.weight_mask_list[layer_index].size(dim=0)):
                num_neurons_current += int(
                    torch.all(torch.eq(self.weight_mask_list[layer_index][row_index, :], 1)).item() and
                    self.bias_mask_list[layer_index][
                        row_index] == 1)
        return num_neurons_current

    def non_zero_params_count(self):
        num_params_nonzero = 0
        for l in range(len(self.weight_mask_list)):
            num_params_nonzero += torch.count_nonzero(self.layers[l].weight) + torch.count_nonzero(
                self.layers[l].bias)
        return num_params_nonzero

    def forward(self, x):
        layer_outputs = []
        x = self.flatten(x)
        for layer, act in zip(self.layers, self.acts): # stops when the smallest iterable (acts) is exhausted
            x = act(layer(x))
            layer_outputs.append(x)
        x = self.layers[-1](x)
        layer_outputs.append(x)
        return layer_outputs


# training & evaluation methods

def evaluate(device, writer, model, train_loader_eval, test_loader_eval, criterion, epoch, batch_idx, test_loss_vector,
             test_accuracy_vector, train_loss_vector, train_accuracy_vector):
    model.eval()
    # evaluation on test set
    test_loss, test_correct = 0, 0
    for data, target in test_loader_eval:
        data = data.to(device)
        target = target.to(device)
        output = model(data)[-1]
        test_loss += criterion(output, target).data.item()
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        test_correct += pred.eq(target.data).cpu().sum().item()

    test_loss /= len(test_loader_eval)
    test_loss_vector.append(test_loss)
    writer.add_scalar("Test_Loss", test_loss, len(train_loader_eval) * (epoch - 1) + batch_idx)

    test_accuracy = 100. * test_correct / len(test_loader_eval.dataset)
    test_accuracy_vector.append(test_accuracy)
    writer.add_scalar("Test_Accuracy", test_accuracy, len(train_loader_eval) * (epoch - 1) + batch_idx)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, test_correct, len(test_loader_eval.dataset), test_accuracy))

    # evaluation on training set
    train_loss, train_correct = 0.0, 0.0
    for data, target in train_loader_eval:
        data = data.to(device)
        target = target.to(device)
        output = model(data)[-1]
        train_loss += criterion(output, target).data.item()
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        train_correct += pred.eq(target.data).cpu().sum().item()

    train_loss /= len(train_loader_eval)
    train_loss_vector.append(train_loss)
    writer.add_scalar("Train_Loss", train_loss, len(train_loader_eval) * (epoch - 1) + batch_idx)

    train_accuracy = 100. * train_correct / len(train_loader_eval.dataset)
    train_accuracy_vector.append(train_accuracy)
    writer.add_scalar("Train_Accuracy", train_accuracy, len(train_loader_eval) * (epoch - 1) + batch_idx)

    print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        train_loss, train_correct, len(train_loader_eval.dataset), train_accuracy))


def train(device, writer, optimizer, model, train_loader, train_loader_eval, test_loader_eval, criterion, epoch,
          test_loss_vector, test_accuracy_vector, train_loss_vector, train_accuracy_vector, num_expansions_vector,
          num_params_vector, epoch_vector, log_interval, num_expansions, debug_grad_flag, gradient_clip):
    # Set model to training mode
    model.train()

    # Loop over each batch from the training set
    for batch_idx, (data, target) in enumerate(train_loader):
        # Copy data to GPU if needed
        data = data.to(device)
        target = target.to(device)

        # Zero gradient buffers
        optimizer.zero_grad()

        # Pass data through the network
        output_layers = model(data)

        # Calculate loss
        loss = criterion(output_layers[-1], target)

        if debug_grad_flag:
            # zero out gradients wrt layer_outputs if needed
            for layer_index in range(len(output_layers)):
                output_layers[layer_index].retain_grad()
                if output_layers[layer_index].grad is not None:
                    output_layers[layer_index].grad.zero_()

        # Backpropagate - gradient computation
        loss.backward()

        if debug_grad_flag:
            grads = [param.grad.detach().flatten() for param in model.parameters() if param.grad is not None]
            norm = torch.cat(grads).norm()
        # print(model.layers[0].weight.grad)
        # zero out gradient of non-existing edges
        for i in range(len(model.layers)):
            model.layers[i].weight.grad *= model.weight_mask_list[i]
            model.layers[i].bias.grad *= model.bias_mask_list[i]

        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)  # clip gradient norm to avoid exploding gradients
        # Update weights
        optimizer.step()

        if batch_idx % log_interval == 0:
            if debug_grad_flag:
                print("----------------------Total gradient norm: ", norm, "-------------------------")
                # print gradients wrt layer_inputs if needed
                # print(model.layers[2].weight)
                # print(output_layers[2].mean(axis=0))
                # for i in range(len(output_layers)):
                #     print(f"Layer {i} weight gradient (mean over batch and matrix entries): ", model.layers[i].weight.grad[:, :].mean())
                #     print(f"Layer {i} output gradient (mean over batch and neurons): ", output_layers[i].grad[:, :].mean())
                    # if i > 0:
                    #     existing_neurons_list = get_existing_neurons_in_layer(i-1, model.weight_mask_list,
                    #                                                           model.bias_mask_list)
                    #     print(f"Layer {i} input (mean over batch and existing neurons): ", output_layers[i-1][:, existing_neurons_list].mean())
                    # else:
                    #     print(f"Layer {i} input (mean over batch and existing neurons): ", data[:, :].mean())

            # logs
            writer.add_scalar("Epoch", epoch, len(train_loader) * (epoch - 1) + batch_idx)
            epoch_vector.append(epoch)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data.item()))
            writer.add_scalar("Batch_Loss", loss.data.item(), len(train_loader) * (epoch - 1) + batch_idx)
            num_params = model.param_count().cpu().item()
            print("current number of expansions: ", num_expansions)
            print("current number of params: ", num_params)
            num_expansions_vector.append(num_expansions)
            num_params_vector.append(num_params)
            writer.add_scalar("Num_Expansions", num_expansions, len(train_loader) * (epoch - 1) + batch_idx)
            writer.add_scalar("Num_Parameters", num_params, len(train_loader) * (epoch - 1) + batch_idx)
            evaluate(device, writer, model, train_loader_eval, test_loader_eval, criterion, epoch, batch_idx,
                     test_loss_vector,
                     test_accuracy_vector, train_loss_vector, train_accuracy_vector)


def initWeight(device, model, layer_index, row_index, column_index, is_weight):
    init_boundary = np.sqrt(
        6 / (model.weight_mask_list[layer_index].size(dim=0) + model.weight_mask_list[layer_index].size(dim=1)))
    if is_weight:
        model.weight_mask_list[layer_index][row_index, column_index] = 1
        with torch.no_grad():
            torch.nn.init.uniform_(model.layers[layer_index].weight[row_index, column_index],
                                   a=-init_boundary,
                                   b=init_boundary)

    else:  # bias param
        model.bias_mask_list[layer_index][row_index] = 1
        with torch.no_grad():
            torch.nn.init.uniform_(model.layers[layer_index].bias[row_index],
                                   a=-init_boundary,
                                   b=init_boundary)


def initNeuron(device, model, layer_index, row_index):
    model.weight_mask_list[layer_index][row_index, :] = torch.ones(
        size=(model.weight_mask_list[layer_index].size(dim=1),), device=device, dtype=torch.int32)
    model.bias_mask_list[layer_index][row_index] = 1
    init_boundary = np.sqrt(
        6 / (model.weight_mask_list[layer_index].size(dim=0) + model.weight_mask_list[
            layer_index].size(
            dim=1)))
    with torch.no_grad():
        torch.nn.init.uniform_(
            model.layers[layer_index].weight[row_index, :],
            a=-init_boundary,
            b=init_boundary)
        torch.nn.init.uniform_(model.layers[layer_index].bias[row_index],
                               a=-init_boundary,
                               b=init_boundary)


def init_all_masked_params(device, model):
    masked_params_idx = get_available_edges_global(model.weight_mask_list, model.bias_mask_list, model.params_total())
    for index in masked_params_idx:
        layer_index, row_index, column_index, is_weight = get_edge_model_indices_from_global_array(index, model.weight_mask_list, model.bias_mask_list)
        initWeight(device, model, layer_index, row_index, column_index, is_weight)

def get_global_index_neuron_array(layer_index, row_index, weight_mask_list):
    result_index = 0
    for l in range(layer_index):
        result_index += weight_mask_list[l].size(dim=0)
    result_index += row_index
    return result_index


def get_neuron_model_indices_from_global_array(array_index, weight_mask_list):
    if torch.is_tensor(array_index):
        array_index = array_index.item()
    layer_index = -1
    row_index = -1
    for l in range(len(weight_mask_list)):
        if array_index - weight_mask_list[l].size(dim=0) < 0:
            layer_index = l
            row_index = array_index
            break
        array_index -= weight_mask_list[l].size(dim=0)
    return layer_index, row_index


def get_global_index_edges_array(layer_index, row_index, column_index, is_weight, weight_mask_list, bias_mask_list):
    result_index = 0
    if is_weight:
        for l in range(layer_index):
            result_index += torch.numel(weight_mask_list[l])
            result_index += torch.numel(bias_mask_list[l])
        result_index += (row_index * weight_mask_list[layer_index].size(dim=1) + column_index)
    else:  # if bias, only the row index is relevant
        for l in range(layer_index):
            result_index += torch.numel(weight_mask_list[l])
            result_index += torch.numel(bias_mask_list[l])
        result_index += (torch.numel(weight_mask_list[layer_index]) + row_index)
    return result_index


def get_edge_model_indices_from_global_array(array_index, weight_mask_list, bias_mask_list):
    if torch.is_tensor(array_index):
        array_index = array_index.item()
    layer_index = -1
    row_index = -1
    column_index = -1
    is_weight = True
    for l in range(len(weight_mask_list)):
        if array_index - torch.numel(weight_mask_list[l]) < 0:
            layer_index = l
            is_weight = True
            row_index, column_index = reconstruct_2d_index_from_flattened(array_index, weight_mask_list[l].size(dim=1))
            break
        array_index -= torch.numel(weight_mask_list[l])
        if array_index - torch.numel(bias_mask_list[l]) < 0:
            layer_index = l
            is_weight = False
            row_index = array_index
            break
        array_index -= torch.numel(bias_mask_list[l])
    return layer_index, row_index, column_index, is_weight


def reconstruct_2d_index_from_flattened(index, num_columns):  # layerwise global index
    column_index = index % num_columns
    row_index = index // num_columns
    return row_index, column_index


def get_flattened_weight_matrix_index(row_index, column_index, num_columns):  # layerwise global index
    return row_index * num_columns + column_index


def get_available_neurons_global(weight_mask_list, bias_mask_list, num_neurons_total):
    result_neuron_list = []
    for i in range(num_neurons_total):
        layer_index, row_index = get_neuron_model_indices_from_global_array(i, weight_mask_list)
        if torch.any(torch.eq(weight_mask_list[layer_index][row_index, :], 0)).item() or \
                bias_mask_list[layer_index][
                    row_index] == 0:
            result_neuron_list.append(i)
    return result_neuron_list  # ascending order


def get_available_edges_global(weight_mask_list, bias_mask_list, num_params_total):
    available_edge_idx = []
    for i in range(num_params_total):
        layer_index, row_index, column_index, is_weight = get_edge_model_indices_from_global_array(i, weight_mask_list,
                                                                                                   bias_mask_list)
        if is_weight:
            if weight_mask_list[layer_index][row_index, column_index] == 0:
                available_edge_idx.append(i)
        else:
            if bias_mask_list[layer_index][row_index] == 0:
                available_edge_idx.append(i)
    return available_edge_idx


def get_available_neurons_in_layer(layer_index, weight_mask_list, bias_mask_list):  # local layer index (=row_index)
    result_neuron_list = []
    for i in range(weight_mask_list[layer_index].size(dim=0)):
        if torch.any(torch.eq(weight_mask_list[layer_index][i, :], 0)).item() or bias_mask_list[layer_index][i] == 0:
            result_neuron_list.append(i)
    return result_neuron_list  # ascending order


def get_edge_index_in_layer_array(layer_index, row_index, column_index, is_weight, weight_mask_list, bias_mask_list):
    if is_weight:
        return get_flattened_weight_matrix_index(row_index, column_index, weight_mask_list[layer_index].size(dim=1))
    else:
        return torch.numel(weight_mask_list[layer_index]) + bias_mask_list[layer_index][row_index]


def get_edge_model_indices_from_layer_array(layer_index, array_index, weight_mask_list, bias_mask_list):
    if array_index - torch.numel(weight_mask_list[layer_index]) < 0:
        row_index, column_index = reconstruct_2d_index_from_flattened(array_index,
                                                                      weight_mask_list[layer_index].size(dim=1))
        is_weight = True
    else:
        row_index = array_index - torch.numel(weight_mask_list[layer_index])
        column_index = -1
        is_weight = False
    return row_index, column_index, is_weight


def get_available_edges_in_layer(layer_index, weight_mask_list, bias_mask_list):
    num_params_in_layer = torch.numel(weight_mask_list[layer_index]) + torch.numel(bias_mask_list[layer_index])
    result_edges_list = []
    for i in range(num_params_in_layer):
        row_index, column_index, is_weight = get_edge_model_indices_from_layer_array(layer_index, i, weight_mask_list,
                                                                                     bias_mask_list)
        if is_weight:
            if weight_mask_list[layer_index][row_index, column_index] == 0:
                result_edges_list.append(i)
        else:
            if bias_mask_list[layer_index][row_index] == 0:
                result_edges_list.append(i)
    return result_edges_list

def get_existing_neurons_global(weight_mask_list, bias_mask_list, num_neurons_total):
    result_neuron_list = []
    for i in range(num_neurons_total):
        layer_index, row_index = get_neuron_model_indices_from_global_array(i, weight_mask_list)
        if torch.any(torch.eq(weight_mask_list[layer_index][row_index, :], 1)).item() or \
                bias_mask_list[layer_index][
                    row_index] == 1:
            result_neuron_list.append(i)
    return result_neuron_list  # ascending order

def get_existing_neurons_in_layer(layer_index, weight_mask_list, bias_mask_list):  # local layer index (=row_index)
    result_neuron_list = []
    for i in range(weight_mask_list[layer_index].size(dim=0)):
        if torch.any(torch.eq(weight_mask_list[layer_index][i, :], 1)).item() or bias_mask_list[layer_index][i] == 1:
            result_neuron_list.append(i)
    return result_neuron_list  # ascending order

def get_available_edges_in_neuron(layer_index, row_index, weight_mask_list, bias_mask_list):
    edge_indices_inside_neuron = []
    for i in range(weight_mask_list[layer_index].size(dim=1)):
        if weight_mask_list[layer_index][row_index, i] == 0:
            edge_indices_inside_neuron.append(i)
    is_bias_masked = (bias_mask_list[layer_index][row_index] == 0)
    return edge_indices_inside_neuron, is_bias_masked


def get_neuron_connectivity_to_output_list(weight_mask_list):
    connectivity_list = [torch.zeros(layer.size(dim=0)).to(torch.bool) for layer in weight_mask_list]  # store for each neuron the connectivity to output

    for layer_index in range(len(weight_mask_list)-1, -1, -1):
        for neuron_index in range(weight_mask_list[layer_index].size(dim=0)):

            if layer_index == len(weight_mask_list)-1:  # output neuron is trivially connected
                connectivity_list[layer_index][neuron_index] = True
            else:
                for successor_index in range(weight_mask_list[layer_index+1].size(dim=0)):
                    if (weight_mask_list[layer_index + 1][successor_index, neuron_index] == 1) and connectivity_list[layer_index + 1][successor_index]:
                        connectivity_list[layer_index][neuron_index] = True
                        break
    return connectivity_list


def get_neuron_connectivity_to_input_list(weight_mask_list):
    connectivity_list = [torch.zeros(layer.size(dim=0)).to(torch.bool) for layer in
                         weight_mask_list]  # store for each neuron the connectivity to output

    for layer_index in range(len(weight_mask_list)):
        for neuron_index in range(weight_mask_list[layer_index].size(dim=0)):
            for predecessor_index in range(weight_mask_list[layer_index].size(dim=1)):
                if weight_mask_list[layer_index][neuron_index, predecessor_index] == 1:
                    if layer_index == 0:
                        connectivity_list[layer_index][neuron_index] = True
                    elif connectivity_list[layer_index - 1][predecessor_index]:
                        connectivity_list[layer_index][neuron_index] = True
                        break
    return connectivity_list

def get_connectivity_score(weight_mask_list, debug_flag=False):  # percentage of params connected to input and output nodes (not considering bias params)
    num_params_current = 0
    for l in range(len(weight_mask_list)):
        num_params_current += torch.count_nonzero(weight_mask_list[l])  # without bias weights

    neuron_connectivity_list_to_input = get_neuron_connectivity_to_input_list(weight_mask_list)
    neuron_connectivity_list_to_output = get_neuron_connectivity_to_output_list(weight_mask_list)

    num_params_connected = 0
    for l in range(len(weight_mask_list)):
        for i in range(weight_mask_list[l].size(dim=0)):
            for j in range(weight_mask_list[l].size(dim=1)):
                if weight_mask_list[l][i, j] == 1:
                    if l == 0:
                        is_connected_to_input = True
                    else:
                        is_connected_to_input = neuron_connectivity_list_to_input[l-1][j]
                    is_connected_to_output = neuron_connectivity_list_to_output[l][i]
                    if is_connected_to_input and is_connected_to_output:
                        num_params_connected += 1
                    if debug_flag:
                        print("param connected to input: ", is_connected_to_input)
                        print("param connected to output: ", is_connected_to_output)
                        print("layer: ", l, " weight: ", i, j, " done, value: ", is_connected_to_input and is_connected_to_output)

    connectivity_score = num_params_connected/num_params_current
    return connectivity_score


# ugly implementation, takes way too long, computes the connectivity repeatedly for all neurons/params
def is_param_connected_to_output(layer_idx, row_idx, column_idx, is_weight, weight_mask_list):
    deq = deque([(layer_idx, row_idx)])
    while len(deq) != 0:  # pop: dfs,  popleft: bfs
        current_neuron = deq.pop()  # DFS
        if current_neuron[0] == (len(weight_mask_list) - 1):
            return True
        for i in range(weight_mask_list[current_neuron[0] + 1].size(dim=0)):
            if weight_mask_list[current_neuron[0] + 1][i, current_neuron[1]] == 1:
                deq.append((current_neuron[0] + 1, i))
    return False

def is_param_connected_to_input(layer_idx, row_idx, column_idx, is_weight, weight_mask_list):
    if not is_weight:
        return False  # bias is not connected to input
    deq = deque([(layer_idx, column_idx)])
    while len(deq) != 0:  # pop: dfs,  popleft: bfs
        current_neuron = deq.pop()  # DFS
        if current_neuron[0] == 0:
            return True
        for j in range(weight_mask_list[current_neuron[0] - 1].size(dim=1)):
            if weight_mask_list[current_neuron[0] - 1][current_neuron[1], j] == 1:
                deq.append((current_neuron[0] - 1, j))
    return False

# def get_connectivity_score(weight_mask_list, debug_flag=False):  # percentage of params connected to input and output nodes (not considering bias params)
#     num_params_current = 0
#     for l in range(len(weight_mask_list)):
#         num_params_current += torch.count_nonzero(weight_mask_list[l])  # without bias weights
#
#     num_params_connected = 0
#     for l in range(len(weight_mask_list)):
#         for i in range(weight_mask_list[l].size(dim=0)):
#             for j in range(weight_mask_list[l].size(dim=1)):
#                 if weight_mask_list[l][i, j] == 1:
#                     is_connected_to_input = is_param_connected_to_input(l, i, j, True, weight_mask_list)
#                     is_connected_to_output = is_param_connected_to_output(l, i, j, True, weight_mask_list)
#                     if is_connected_to_input and is_connected_to_output:
#                         num_params_connected += 1
#                     if debug_flag:
#                         print("param connected to input: ", is_connected_to_input)
#                         print("param connected to output: ", is_connected_to_output)
#                         print("layer: ", l, " weight: ", i, j, " done, value: ", is_connected_to_input and is_connected_to_output)
#
#     connectivity_score = num_params_connected/num_params_current
#     return connectivity_score

