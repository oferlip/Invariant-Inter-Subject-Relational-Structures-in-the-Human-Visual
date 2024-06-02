from typing import List, Tuple
import os
import numpy as np
import pandas as pd
from itertools import repeat, combinations
from datetime import datetime
import copy
from scipy.spatial.distance import cdist
from dataclasses import dataclass
from copy import deepcopy
from scipy import stats
from scipy.optimize import linear_sum_assignment
import random
from multiprocessing.pool import ThreadPool
import itertools
import json
from typing import Callable
import math
import sys
from operator import itemgetter


absolute_path_to_repository = "Insert path to repository"
sys.path.append(os.path.dirname(f"{absolute_path_to_repository}/code/similarity/"))
sys.path.append(os.path.dirname(f"{absolute_path_to_repository}/code/"))
sys.path.append(os.path.dirname(absolute_path_to_repository))

from utils import numpy_utils
from data_stucture.BrainData import BrainData, ContactKind, ImageKind
from results_code.result_cache_utils import write_date, MaskValues, write_metadata, write_csv



@dataclass
class MeasurementConfig:
    create_trnasfomraiton_method: Callable[[np.array, np.array], np.array]
    create_trnasfomraiton_method_name: str



from scipy.stats import spearmanr

def get_columns_correlation_mean(matrix1, matrix2):
  return spearmanr(matrix1, matrix2, axis=0)[0]






def nCr(n,r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)


def align_features_distances(should_normalize_contacts, should_normalize_images, should_take_features=True):
    def distances_alignment_helper(features_subject_1, features_subject_2, contacts_xyz_subject_1, contacts_xyz_subject_2):

        # distances alignment
        # normalized
        if should_take_features:
            if should_normalize_images:
                if should_normalize_images:
                    features_subject_1 = numpy_utils.normalize_instances(features_subject_1)
                    features_subject_2 = numpy_utils.normalize_instances(features_subject_2)

            features_subject_1_transformed, features_subject_2_transformed = features_subject_1.T, features_subject_2.T
            if should_normalize_contacts:
                features_subject_1_transformed = numpy_utils.normalize_instances(features_subject_1_transformed)
                features_subject_2_transformed = numpy_utils.normalize_instances(features_subject_2_transformed)
            consts = cdist(features_subject_1_transformed, features_subject_2_transformed)
        else:
            if should_normalize_images or should_normalize_contacts:
                raise
            consts = cdist(contacts_xyz_subject_1, contacts_xyz_subject_2)


        row_ind, col_ind = linear_sum_assignment(consts, maximize=False)
        return features_subject_1[:, row_ind], features_subject_2[:, col_ind], row_ind, col_ind
    return distances_alignment_helper

def get_align_features_spearman_method(should_normalize_images, should_normalize_contacts):
    def align_features_spearman(features_subject_1, features_subject_2, xyz_subject_1, xyz_subject_2s):

        if should_normalize_images:
            features_subject_1 = numpy_utils.normalize_instances(features_subject_1)
            features_subject_2 = numpy_utils.normalize_instances(features_subject_2)
        if should_normalize_contacts:
            features_subject_1 = numpy_utils.normalize_instances(features_subject_1.T).T
            features_subject_2 = numpy_utils.normalize_instances(features_subject_2.T).T

        spearman_values = stats.spearmanr(features_subject_1, features_subject_2, axis=0)[0]

        if isinstance(spearman_values, float):
            row_ind, col_ind = np.array([0]), np.array([0])

        else:
            consts = spearman_values[: features_subject_1.shape[1], features_subject_1.shape[1]:]
            row_ind, col_ind = linear_sum_assignment(consts, maximize=True)


        return features_subject_1[:, row_ind], features_subject_2[:, col_ind], row_ind, col_ind

    return align_features_spearman


def get_align_features_spearman_method_1_side(features_subject_1, features_subject_2, xyz_subject_1, xyz_subject_2s):



    spearman_values = stats.spearmanr(features_subject_1, features_subject_2, axis=0)[0]

    if isinstance(spearman_values, float): # features_subject_2 = np.random.random((10, 5))
        row_ind, col_ind = np.array([0]), np.array([0])

    else:
        consts = spearman_values[: features_subject_1.shape[1], features_subject_1.shape[1]:]
        row_ind, col_ind = linear_sum_assignment(consts, maximize=True)

    reordered_features_subject_1, reordered_features_subject_2 = np.copy(features_subject_1[:, row_ind]), np.copy(features_subject_2[:, col_ind])
    for i in range(reordered_features_subject_2.shape[1]):
        reordered_features_subject_1[:, i] = reordered_features_subject_1[:, i] * get_row_best_match(reordered_features_subject_1[:, i], reordered_features_subject_2[:, i])

    return reordered_features_subject_1, reordered_features_subject_2, row_ind, col_ind


def get_row_best_match(a, b):
    return (b @ a.T) / (a @ a.T)


def align_features_pearson(features_subject_1, features_subject_2, xyz_1, xyz_2):
    consts = np.corrcoef(features_subject_1.T, features_subject_2.T)[:features_subject_1.shape[1], features_subject_1.shape[1]:]
    row_ind, col_ind = linear_sum_assignment(consts, maximize=True)

    return features_subject_1[:, row_ind], features_subject_2[:, col_ind], row_ind, col_ind


def findsubsets(s: set, n: int):
    list_test_combinations = []
    nuber_of_combinations = nCr(len(s), n)
    max_number_of_combinations = min (100, int(nuber_of_combinations))
    if nuber_of_combinations > 1000:
        for i in range(max_number_of_combinations):
            while True:
                current_combination_train = set(sorted(random.sample(s, n)))
                if current_combination_train not in list_test_combinations:
                    list_test_combinations.append(current_combination_train)
                    break
    else:
        list_test_combinations = random.choices(list(itertools.combinations(s, int(n))), k=max_number_of_combinations)


    list_train_combinations = [s - set(current_combination) for current_combination in list_test_combinations]
    return list_test_combinations, list_train_combinations



def create_mask(features, rows_in_mask):
    mask = np.zeros(features.shape)
    mask[np.array(sorted(list(rows_in_mask))), :] = 1
    return mask

def create_masks(first_subject_features, list_test_combinations, list_train_combinations):
    return [MaskValues(test_mask=create_mask(first_subject_features, list_test_combinations[i]),
                train_mask=create_mask(first_subject_features, list_train_combinations[i]))
     for i in range(len(list_test_combinations))]

def get_masks(first_subject_features, second_subject_features, size_of_test_group=None) -> List[MaskValues]:
    features_shape = first_subject_features.shape
    number_of_images = features_shape[0]
    list_test_combinations, list_train_combinations = findsubsets(set(np.arange(start=0, stop=number_of_images, step=1)), size_of_test_group)
    return create_masks(first_subject_features, list_test_combinations, list_train_combinations)



def create_subject_to_subject_data(subjects_features,
                                   subjects_xyz_coordintes,
                                   subjects_contacts_hamesphare,
                                   subject_indexes,
                                   create_transformation_matrix,
                                   create_transformation_matrix_name,
                                   contact_name,
                                   image_kinds,
                                   database_name,
                                   min_number_of_contacts,
                                   subjects_contacts_types):
    subject_1_index, subject_2_index = subject_indexes
    
    features_subject_1 = subjects_features[subject_1_index]
    features_subject_2 = subjects_features[subject_2_index]

    subject1_xyz_coordinates = subjects_xyz_coordintes[subject_1_index]
    subject2_xyz_coordinates = subjects_xyz_coordintes[subject_2_index]

    features_subject_1, features_subject_2, features_subject_1_indexes, features_subject_2_indexes = align_features(features_subject_1,
                                                                                                                    features_subject_2,
                                                                                                                    subject1_xyz_coordinates,
                                                                                                                    subject2_xyz_coordinates)

    subject1_contacts_hamesphare = subjects_contacts_hamesphare[subject_1_index]
    subject2_contacts_hamesphare = subjects_contacts_hamesphare[subject_2_index]

    subject1_contacts_types = subjects_contacts_types[subject_1_index]
    subject2_contacts_types = subjects_contacts_types[subject_2_index]

    subject1_xyz_coordinates, subject2_xyz_coordinates = subject1_xyz_coordinates[features_subject_1_indexes, :], subject2_xyz_coordinates[features_subject_2_indexes, :]


    subject1_contacts_hamesphare = itemgetter(*features_subject_1_indexes)(subject1_contacts_hamesphare)
    subject2_contacts_hamesphare = itemgetter(*features_subject_2_indexes)(subject2_contacts_hamesphare)

    subject1_contacts_types = itemgetter(*features_subject_1_indexes)(subject1_contacts_types)
    subject2_contacts_types = itemgetter(*features_subject_2_indexes)(subject2_contacts_types)

    for size_of_test_group in range(2, 3, 1):
        masks = get_masks(features_subject_1, features_subject_2, size_of_test_group)

        transformation_matrices = []

        for i, current_masks in enumerate(masks):
            masked_features_1 = features_subject_1[np.array(current_masks.train_mask, dtype=bool)].reshape(-1, features_subject_1.shape[1])
            masked_features_2 = features_subject_2[np.array(current_masks.train_mask, dtype=bool)].reshape(-1, features_subject_2.shape[1])
           
            transformation_matrix = create_transformation_matrix(masked_features_1, masked_features_2)
            transformation_matrices.append(transformation_matrix)
#

        write_date(output_folder_path,
                   subject_1_index,
                   subject_2_index,
                   subject1_xyz_coordinates,
                   subject2_xyz_coordinates,
                   subject1_contacts_hamesphare,
                   subject2_contacts_hamesphare,
                   features_subject_1,
                   features_subject_2,
                   features_subject_1_indexes,
                   features_subject_2_indexes,
                   transformation_matrices,
                   masks,
                   database_name,
                   contact_name,
                   image_kinds,
                   create_transformation_matrix_name,
                   min_number_of_contacts,
                   subject1_contacts_types,
                   subject2_contacts_types,
                   size_of_test_group)

def create_image_kinds_data_files(current_measurement_configions: List[MeasurementConfig],
                                  brain_data: BrainData,
                                  images_kinds: List[ImageKind],
                                  contact_kinds_to_test: List[ContactKind]):
    for contact_kind_to_test in contact_kinds_to_test:
        print("contact_kind_to_test: ", str(contact_kind_to_test), ", with images kinds: ", str(images_kinds))
        min_number_of_contacts = 4

        subjects_data = brain_data.get_brain_level_feature_image(contact_type=contact_kind_to_test,
                                                                         make_feature_list=True,
                                                                         use_people_with_high_and_low_contacts=False,
                                                                         images_kinds=images_kinds)
        subjects_data = brain_data.get_brain_level_feature_image(contact_type=contact_kind_to_test,
                                                                 make_feature_list=True,
                                                                 use_people_with_high_and_low_contacts=False,
                                                                 images_kinds=images_kinds)
       
        print("finished importing data")
        subjects_xyz_coordinates = brain_data.get_xyz_coordinates_all_subjects(contact_kind_to_test)
        subjects_contacts_hamesphare = brain_data.get_contacts_hamesphare_all_subjects(contact_kind_to_test)
        subjects_contacts_types = brain_data.get_all_subject_contacts_types(contact_kind_to_test)

        subjects__high_contact_kind_indexes = []
        for i, cur_subject_name in enumerate(brain_data.sorted_subects_names):
            current_subject_contacts = []
            for j, curr_contact_name in enumerate(brain_data.subject_to_electrodes[cur_subject_name]):
                if brain_data.subject_to_electrode_to_contact_type[cur_subject_name, curr_contact_name] == ContactKind.High:
                    current_subject_contacts.append(j)
            subjects__high_contact_kind_indexes.append(current_subject_contacts)

        subjects_data_indexes = [i for i, current_subject_data in
                                 enumerate(subjects_data) if
                                  current_subject_data.shape[1] >= min_number_of_contacts and len(subjects__high_contact_kind_indexes[i]) >= min_number_of_contacts]



        subjects_data_cpy = copy.deepcopy(subjects_data)
        subjects_data = [current_subject_data for i, current_subject_data in enumerate(subjects_data) if i in subjects_data_indexes]
        subjects_xyz_coordinates_cpy = copy.deepcopy(subjects_xyz_coordinates)
        subjects_xyz_coordinates = [current_subject_xyz_coordinates for i, current_subject_xyz_coordinates in enumerate(subjects_xyz_coordinates) if i in subjects_data_indexes]
        subjects_contacts_hamesphare = [current_subjects_contacts_hamesphare for i, current_subjects_contacts_hamesphare in enumerate(subjects_contacts_hamesphare) if i in subjects_data_indexes]
        subjects_contacts_types = [current_subjects_contacts_types for i, current_subjects_contacts_types in enumerate(subjects_contacts_types) if i in subjects_data_indexes]


        for measurement_config in current_measurement_configions:
            write_metadata(output_folder_path,
                           len(subjects_data),
                           brain_data.dataset_name,
                           contact_kind_to_test,
                           images_kinds,
                           measurement_config.create_trnasfomraiton_method_name,
                           min_number_of_contacts)


            pairs_combinations = [(index_1, index_2) for index_1 in list(range(len(subjects_data))) for index_2 in
                                  list(range(len(subjects_data)))]
            ii = 0
            while ii < 1:
                try:
                    with ThreadPool(processes=5) as pool:
                        pool.starmap(create_subject_to_subject_data, zip(repeat(subjects_data),
                                                                         repeat(subjects_xyz_coordinates),
                                                                         repeat(subjects_contacts_hamesphare),
                                                                         pairs_combinations,
                                                                         repeat(measurement_config.create_trnasfomraiton_method),
                                                                         repeat(measurement_config.create_trnasfomraiton_method_name),
                                                                         repeat(contact_kind_to_test),
                                                                         repeat(images_kinds),
                                                                         repeat(brain_data.dataset_name),
                                                                         repeat(min_number_of_contacts),
                                                                         repeat(subjects_contacts_types)))


                        ii += 1

                except:
                    ii += 1
                    raise


def crete_csv_used_contacts(used_indexes_sets,
                            subjects_xyz_coordinates_cpy,
                            brain_data,
                            subjects_data_indexes,
                            current_contact_kind,
                            image_kinds,
                            min_number_of_contacts):
    csv_data = []
    subjects_xyz_coordinates = brain_data.get_xyz_coordinates_all_subjects(ContactKind.NoneAbove)
    for index in used_indexes_sets:

        curernt_subject_used_indexes = used_indexes_sets[index]
        curr_subject = brain_data.sorted_subects_names[index]
        cur_subject_electrodes = brain_data.subject_to_electrodes[curr_subject]
        elecrode_used = [elec for i, elec in enumerate(cur_subject_electrodes) if i in curernt_subject_used_indexes]
        subjects_xyz_coordinates_used = [xyz for i, xyz in enumerate(subjects_xyz_coordinates[index]) if i in curernt_subject_used_indexes]
        csv_data += list(zip(repeat([curr_subject]), elecrode_used, subjects_xyz_coordinates_used))
    df = pd.DataFrame(csv_data, columns=["subject", "electrode", "xyz"])
    write_csv(output_folder_path,df,curent_database_name, current_contact_kind, image_kinds,min_number_of_contacts)

def create_subject_to_subject_map_of_used_electrdes(subject_indexes,
                                                    subjects_data_cpy,
                                                    subjects_xyz_coordintes,
                                                    subjects_data_indexes_tmp,
                                                    subjects_high_contact_kind_indexes,
                                                    used_indexes_sets):
    subject_1_index, subject_2_index = subjects_data_indexes_tmp[subject_indexes[0]], subjects_data_indexes_tmp[subject_indexes[1]]
    if subject_1_index == subject_2_index:
        return

    features_subject_1 = subjects_data_cpy[subject_1_index]
    features_subject_2 = subjects_data_cpy[subject_2_index]
    features_subject_1_high_contact_indexes = subjects_high_contact_kind_indexes[subject_1_index]
    features_subject_2_high_contact_indexes = subjects_high_contact_kind_indexes[subject_2_index]

    subject1_xyz_coordinates = subjects_xyz_coordintes[subject_indexes[0]]
    subject2_xyz_coordinates = subjects_xyz_coordintes[subject_indexes[1]]

    features_subject_1, features_subject_2, features_subject_1_indexes, features_subject_2_indexes = align_features(
        features_subject_1, features_subject_2, subject1_xyz_coordinates, subject2_xyz_coordinates)

    used_indexes_sets[subject_1_index] = used_indexes_sets[subject_1_index].union([features_subject_1_high_contact_indexes[ind] for ind in features_subject_1_indexes])
    used_indexes_sets[subject_2_index] = used_indexes_sets[subject_2_index].union([features_subject_2_high_contact_indexes[ind] for ind in features_subject_2_indexes])


def get_aligned_rotated_features_measurement_method(should_normalize_instances=False):
    def find_best_orthogonal_matrix_using_svd(B_a, B_b):
        # implement algorithm from here:
        # https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem#Generalized/constrained_Procrustes_problems

        if should_normalize_instances:
            B_a = numpy_utils.normalize_instances(B_a)
            B_b = numpy_utils.normalize_instances(B_b)

        M = B_a.T @ B_b
        U, S, Vh = np.linalg.svd(M, full_matrices=False)
        R = U @ Vh
        return R

    create_trnasfomraiton_method_name = "orthogonal"

    return MeasurementConfig(create_trnasfomraiton_method=find_best_orthogonal_matrix_using_svd,
                             create_trnasfomraiton_method_name=create_trnasfomraiton_method_name)

def get_aligned_linear_measurement_method(should_normalize=False):
    def find_tranformation_matrix_pseudo_inverse(B_a, B_b):
        if should_normalize:
            B_a = numpy_utils.normalize_instances(B_a)
            B_b = numpy_utils.normalize_instances(B_b)
        return np.linalg.pinv(B_a) @ B_b

    create_trnasfomraiton_method_name = "linear"
    if should_normalize:
        create_trnasfomraiton_method_name += "_normalized"
    return MeasurementConfig(create_trnasfomraiton_method=find_tranformation_matrix_pseudo_inverse,
                             create_trnasfomraiton_method_name=create_trnasfomraiton_method_name)


def create_data_files_all_kinds(current_measurement_configions: List[MeasurementConfig],
                                brain_data: BrainData,
                                contact_kinds_to_test: List[ContactKind],
                                curent_database_name):
    all_images_kinds = {
        "eventrelatednatural": [
                                [ImageKind.All],
                                [ImageKind.Animals],
                                [ImageKind.Patterns],
                                [ImageKind.People],
                                [ImageKind.Places],
                                [ImageKind.Tools]
        ],
        "blockdesign": [
                        [ImageKind.All],
                        [ImageKind.Body],
                        [ImageKind.Face],
                        [ImageKind.House],
                        [ImageKind.Patterns],
                        [ImageKind.Tool]

                        ],
        "eventrelatedold": [
                            [ImageKind.All],
                            [ImageKind.Face],
                            [ImageKind.House],
                            [ImageKind.Tool]
                            ]
    }

    images_kinds = all_images_kinds[curent_database_name]
    for image_kinds in images_kinds:
        print("current image_kinds: ", str(image_kinds))
        create_image_kinds_data_files(current_measurement_configions, brain_data, image_kinds, contact_kinds_to_test)
        pass





###########

alignment_proerties = [
                       # ("distances_alignment", align_features_distances(should_normalize_contacts=False, should_normalize_images=False)),
                       # ("coordinate_distances_alignment", align_features_distances(should_normalize_contacts=False, should_normalize_images=False, should_take_features=False)),
                       # ("distances_normalized_alignment", align_features_distances(should_normalize=True)),
                       # ("pearson_alignment", align_features_pearson),
                        ("spearman_alignment", get_align_features_spearman_method(should_normalize_images=False, should_normalize_contacts=False)),
                       #  ("spearman_alignment_normalize_images", get_align_features_spearman_method(should_normalize_images=True, should_normalize_contacts=False)),
                       #  ("spearman_alignment_normalize_contacts", get_align_features_spearman_method(should_normalize_images=False, should_normalize_contacts=True)),
                       #  ("spearman_alignment_normalize_2_side_of_contacts", get_align_features_spearman_method_1_side)
]

databases_name = ["eventrelatednatural", "eventrelatedold", "blockdesign"]
contacts_to_test = [ContactKind.High, ContactKind.All]
contacts_to_test = [ContactKind.All]

if __name__ == '__main__':
    starting_time = datetime.now()
    for current_alignment_proerties in alignment_proerties:
        align_features = current_alignment_proerties[1]
        alignment_name = current_alignment_proerties[0]
        print("starting alignment: " + alignment_name)

        for curent_database_name in databases_name:
            print("current_database " + curent_database_name)
            current_brain_data = BrainData(curent_database_name)
            
            output_folder_path = alignment_name + "/"

            measurement_configurations = [
                get_aligned_rotated_features_measurement_method(should_normalize_instances=False),
                get_aligned_linear_measurement_method(should_normalize=False),
            ]

            create_data_files_all_kinds(measurement_configurations, current_brain_data, contacts_to_test, curent_database_name)

        print("finish running " + str(datetime.now() - starting_time))