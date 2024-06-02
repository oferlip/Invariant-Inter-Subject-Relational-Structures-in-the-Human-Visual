import numpy as np
from sklearn import preprocessing
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import scipy.stats as stats
import warnings



def normalize_instances(data):
    return preprocessing.normalize(data)

def scramble(a, axis=-1):
    """
    Return an array with the values of `a` independently shuffled along the
    given axis
    """
    b = a.swapaxes(axis, -1)
    n = a.shape[axis]

    idx = np.random.choice(n, n, replace=False)
    b = b[..., idx]
    return b.swapaxes(axis, -1)

def shuffle_entire_array(arr):
    arr_shape = arr.shape
    arr = np.copy(arr)
    arr = arr.ravel()
    np.random.shuffle(arr)
    return arr.reshape(arr_shape)

def shuffle_features(features, axis=0, should_shuffle_axis_individual=False):
    shuffled_features = np.copy(features)
    if axis == 1:
        shuffled_features = shuffled_features.T

    if should_shuffle_axis_individual:
        for i in range(shuffled_features.shape[0]):
            np.random.shuffle(shuffled_features[i, :])
    else:
        np.random.shuffle(shuffled_features)

    if axis == 1:
        shuffled_features = shuffled_features.T
    return shuffled_features

def shuffle_random_feature_list(features_list, axis=0, should_shuffle_axis_individual=False):
    result = []
    for features in features_list:
        result.append(shuffle_features(features, axis, should_shuffle_axis_individual))

    return result


def get_examples_best(triangle_matrix, number_of_examples_to_return):
    indexes = np.unravel_index(np.argsort(triangle_matrix.ravel())[-number_of_examples_to_return:],
                     triangle_matrix.shape)
    return indexes

def take_off_diagonal_values(mat):
    off_diagonal_indices_a = np.where(~np.eye(mat.shape[0], dtype=bool))
    return mat[off_diagonal_indices_a]

# worst fit
def get_examples_worst(triangle_matrix, number_of_examples_to_return):
    triangle_matrix = triangle_matrix.copy()
    triangle_matrix += np.triu(np.ones(triangle_matrix.shape) * 2)
    indexes = np.unravel_index(np.argsort(triangle_matrix.ravel())[:number_of_examples_to_return],
                     triangle_matrix.shape)
    return indexes

#  middle
def get_examples_mid(triangle_matrix, number_of_examples_to_return):
    indexes_arr = np.argsort(triangle_matrix.ravel())
    starting_index = np.sum(np.triu(np.ones(triangle_matrix.shape)))
    mid_index = (indexes_arr.shape[0] - starting_index) / 2 + starting_index

    indexes = np.unravel_index(indexes_arr[int(mid_index - number_of_examples_to_return / 2) :
                                           int(mid_index + number_of_examples_to_return / 2)],
                               triangle_matrix.shape)
    return indexes


def normal_distances_correlation_method(features_1, features_2, should_normalize=False):
    if len(features_1.shape) == 1:
        features_1 = features_1.reshape(1, -1)
        features_2 = features_2.reshape(1, -1)
    if should_normalize:
        features_1 = preprocessing.normalize(features_1)
        features_2 = preprocessing.normalize(features_2)

    distances_matrix = cdist(features_1, features_2)
    return np.mean(np.diag(distances_matrix))


def normal_cosine_correlation_method(features_1, features_2, should_normalize=False):
    if len(features_1.shape) == 1:
        features_1 = features_1.reshape(1, -1)
        features_2 = features_2.reshape(1, -1)
    if should_normalize:
        features_1 = preprocessing.normalize(features_1)
        features_2 = preprocessing.normalize(features_2)
    return np.mean(np.diag(features_1 @ features_2.T))

def cosine_matrix(features_1, features_2, should_normalize=False):
    if len(features_1.shape) == 1:
        features_1 = features_1.reshape(1, -1)
        features_2 = features_2.reshape(1, -1)
    if should_normalize:
        features_1 = preprocessing.normalize(features_1)
        features_2 = preprocessing.normalize(features_2)
    return features_1 @ features_2.T

def distances_matrix(features_1, features_2, should_normalize=False):
    if len(features_1.shape) == 1:
        features_1 = features_1.reshape(1, -1)
        features_2 = features_2.reshape(1, -1)
    if should_normalize:
        features_1 = preprocessing.normalize(features_1)
        features_2 = preprocessing.normalize(features_2)

    return cdist(features_1, features_2)

def pearson_matrix(features_1, features_2, should_normalize=False):
    if len(features_1.shape) == 1:
        features_1 = features_1.reshape(1, -1)
        features_2 = features_2.reshape(1, -1)
    if should_normalize:
        features_1 = preprocessing.normalize(features_1)
        features_2 = preprocessing.normalize(features_2)

    return np.corrcoef(features_1, features_2)[:features_1.shape[0], features_1.shape[0]:]

def spearman_matrix(features_1, features_2, should_normalize=False):
    if len(features_1.shape) == 1:
        features_1 = features_1.reshape(1, -1)
        features_2 = features_2.reshape(1, -1)
    if should_normalize:
        features_1 = preprocessing.normalize(features_1)
        features_2 = preprocessing.normalize(features_2)

    if features_1.shape[0] == 1:
        return np.array([stats.spearmanr(features_1, features_2, axis=1)[0]])

    return stats.spearmanr(features_1, features_2, axis=1)[0][:features_1.shape[0], features_1.shape[0]:]




# def align_features_to_nearest(correlation_method, take_max=True):
#     def align_features_to_nearest_helper(features_1, features_2):
#         if features_1.shape[1] > features_2.shape[1]:
#             features_2, features_1 = features_1, features_2
#
#         num_features_1 = features_1.shape[1]
#         num_features_2 = features_2.shape[1]
#
#         used_indexes_features_2 = []
#         unused_indexes_features_2 = [i for i in range(num_features_2)]
#         for i in range(num_features_1):
#             contact_i_correlations = []
#             for j in unused_indexes_features_2:
#                 contact_i_correlations.append((correlation_method(features_2[:, j], features_1[:, i]), j))
#             try:
#                 sorted_correlations_indexes = sorted(contact_i_correlations, key=lambda x: x[0])
#             except:
#                 a = 3
#                 sorted_correlations_indexes = sorted(contact_i_correlations, key=lambda x: x[0])
#
#             if take_max:
#                 best_index = sorted_correlations_indexes[-1][1]
#             else:
#                 best_index = sorted_correlations_indexes[0][1]
#             used_indexes_features_2.append(best_index)
#             unused_indexes_features_2.remove(best_index)
#
#         return features_1, features_2[:, used_indexes_features_2]
#         # features_2 = features_2[]
#     return align_features_to_nearest_helper

def align_features_to_nearest(costs, features_1, features_2):
    row_ind, col_ind = linear_sum_assignment(costs)
    return features_1[:, row_ind], features_2[:, col_ind], row_ind, col_ind



def cosine_similarity(x, y):
    x = x / np.linalg.norm(x)
    y = y / np.linalg.norm(y)
    return x @ y.T


def align_features_to_nearest_spearman(B_a, B_b):
    return align_features_to_best_correlation(B_a, B_b, lambda x, y: stats.spearmanr(x,y, axis=0)[0])

def align_features_to_nearest_pearson(B_a, B_b):
    return align_features_to_best_correlation(B_a, B_b, lambda x, y: np.corrcoef(x.T, y.T))

def align_features_to_best_correlation(B_a, B_b, correlation):
    similarity = correlation(B_a, B_b)[:B_a.shape[1], B_a.shape[1]:]

    costs = -similarity + np.max(similarity)
    return align_features_to_nearest(costs, B_a, B_b)

def align_features_to_nearest_distance(B_a, B_b, normalize_matrices=False):
    if normalize_matrices:
        B_a = B_a / np.linalg.norm(B_a)
        B_b = B_b / np.linalg.norm(B_b)
    costs = distances_matrix(B_a.T, B_b.T)
    return align_features_to_nearest(costs, B_a, B_b)

def align_features_to_nearest_similarity(B_a, B_b):
    similarity = cosine_matrix(B_a.T, B_b.T, should_normalize=True)
    costs = -similarity + np.max(similarity)
    return align_features_to_nearest(costs, B_a, B_b)

# a = a = np.arange(4*9).reshape(4, 9)
# array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8],
#        [ 9, 10, 11, 12, 13, 14, 15, 16, 17],
#        [18, 19, 20, 21, 22, 23, 24, 25, 26],
#        [27, 28, 29, 30, 31, 32, 33, 34, 35]])

# print(a)
# print(shuffle_random_feature_list([a], axis=0, should_shuffle_individual=True))
