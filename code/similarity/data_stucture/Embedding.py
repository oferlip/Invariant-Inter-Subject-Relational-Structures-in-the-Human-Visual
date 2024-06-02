from ..utils import embeddings_utils
import numpy as np


class Embeddings:
    def __init__(self, model_name):
        self.model_name = model_name
        self.layer_to_dimentions = {}

    def get_model_name(self):
        return self.model_name

    def get_layer_dimentions(self, layer):

        if layer not in self.layer_to_dimentions:
            self.layer_to_dimentions[layer] = embeddings_utils.get_layer_embeddings(
                self.model_name, layer).shape[1]

        return self.layer_to_dimentions[layer]

    def get_layer_features(self, layer, images_to_remain, features_to_remove, removed_images=None, removed_featues=None):
        embeddings = embeddings_utils.get_layer_embeddings(
            self.model_name, layer, images_to_remain)
        if removed_featues == None:
            embeddings, removed_featues = embeddings_utils.remove_random_weights(
                embeddings, features_to_remove)
        else:
            embeddings = np.delete(embeddings, removed_featues, axis=1)

        if removed_images == None:
            embeddings, removed_images = embeddings_utils.remove_random_examples(
                embeddings, images_to_remain)
        else:
            embeddings = np.delete(embeddings, removed_images, axis=0)

        return embeddings, removed_images, removed_featues

    def get_number_layers(self):
        return embeddings_utils.get_network_embeddings_dimention(self.model_name)
