import os
from ..definitions import definitions
import random
import numpy as np


def get_layer_embeddings_file_name(netwrok_name, layer_index):
    return get_net_embeddings_prefix(netwrok_name) + str(layer_index) + ".npy"


def get_embeddings_file_path_from_full_name(netwrok_name, layer_index):
    return os.path.join(definitions.embeddings_path, netwrok_name + "_layer" + str(layer_index))


def get_net_embeddings_prefix(netwrok_name):
    return os.path.join(definitions.embeddings_path, netwrok_name + "_layer")


def get_network_embeddings_dimention(net_name):
    net_name_prefix = get_net_embeddings_prefix(
        net_name)
    files = [f for f in os.listdir(
        definitions.embeddings_path) if f.startswith(net_name)]
    return len(files)


def remove_random_weights(feature_matrix, persentage_to_remove):
    number_of_cols = feature_matrix.shape[1]
    number_of_cols_to_remove = int(number_of_cols * persentage_to_remove)
    cols_to_remove = random.sample(
        range(number_of_cols), number_of_cols_to_remove)
    return np.delete(feature_matrix, cols_to_remove, axis=1), cols_to_remove


def remove_random_examples(feature_matrix, amount_of_images_to_remain, rows_to_remove=[]):
    number_of_rows = feature_matrix.shape[0]
    number_of_rows_to_remove = number_of_rows - amount_of_images_to_remain
    if len(rows_to_remove) == 0:
        rows_to_remove = random.sample(
            range(number_of_rows), number_of_rows_to_remove)
    return np.delete(feature_matrix, rows_to_remove, axis=0), rows_to_remove


def get_layer_embeddings(network_name, layer, amount_of_images_to_remain, rows_to_remove = []):
    file_name = get_layer_embeddings_file_name(network_name, layer)
    features = np.load(file_name)
    if len(rows_to_remove) != [] or amount_of_images_to_remain != -1:
        features, rows_to_remove = remove_random_examples(features, amount_of_images_to_remain, rows_to_remove)
    return features, rows_to_remove
