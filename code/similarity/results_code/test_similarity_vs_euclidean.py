from typing import List, Tuple
import os
import numpy as np
from itertools import repeat, combinations
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
from json import JSONEncoder
from typing import Callable
import math
from similarity.utils import seaborn_utils, model_utils
import typing
from statistics import mean, variance
from datetime import datetime
from collections import defaultdict
import itertools
from sklearn.manifold import TSNE
from statsmodels.stats.weightstats import ztest as ztest
from scipy.stats import ttest_ind
from scipy.stats import sem, pearsonr
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from scipy import stats

# stats.spearmanr(features_subject_1, features_subject_2, axis=0)[0]

from similarity.data_stucture.BrainData import ContactKind, ImageKind, BrainData
from similarity.results_code.result_cache_utils import read_data, read_metadata
from similarity.utils import numpy_utils


# from sklearn.decomposition import PCA

from similarity.results_code.create_data_files import get_align_features_spearman_method, align_features_pearson



align_features_spearman = get_align_features_spearman_method(should_normalize_images=False, should_normalize_contacts=False)

transformation_name_to_expression = {
    "permutation": "patterns",
    "orthogonal": "relations",
    "linear": "topology"
}

alignment_to_expression = {
    "distances_alignment": "euclidean",
    "coordinate_distances_alignment": "anatomy",
    "spearman_alignment": "optimal"
}

dataset_name_to_code = {
    "eventrelatednatural": "set1",
    "eventrelatedold": "set2",
    "blockdesign": "set3"
}


@dataclass
class GraphConfig:
    transformation_name: str
    neuroscienec_expression: str


all_images_kinds = {
    "eventrelatednatural": [
        # [ImageKind.All],
                            [ImageKind.Animals],
                            [ImageKind.Patterns],
                            [ImageKind.People],
                            [ImageKind.Places],
                            [ImageKind.Tools]
                            ],
    "blockdesign": [[ImageKind.All],
                    [ImageKind.Body],
                    [ImageKind.Face],
                    [ImageKind.House],
                    [ImageKind.Patterns],
                    [ImageKind.Tool]
                    ],
    "eventrelatedold": [[ImageKind.All],
                        [ImageKind.Face],
                        [ImageKind.House],
                        [ImageKind.Tool]
                        ]
}


def create_graph_files_all_kinds():
    min_number_of_contacts = 4

    # TODO: run cross dataset alignment method evaluation
    # create_bar_plot_cross_dataset(min_number_of_contacts)

    #TODO: run contact evalueation
    # create_bar_plot_check_contacts_similarity_cross_dataset(min_number_of_contacts)

    #TODO: run bar_size_of_test_group
    # create_bar_similarity_vs_test_group(min_number_of_contacts)

    #TODO: create subjects activation pattens
    # create_bar_subject_activation_pattern(min_number_of_contacts)

    create_low_high_match_statistic(min_number_of_contacts)



def create_low_high_match_statistic(min_number_of_contacts):
    modified_all_images_kinds = {
        "eventrelatednatural": [
            [ImageKind.All],
            [ImageKind.Animals],  # 0
            [ImageKind.Patterns],  # 1
            [ImageKind.People],  # 2
            [ImageKind.Places],  # 3
            [ImageKind.Tools]],  # 4
        "blockdesign": [
            [ImageKind.Body],  # 8
            [ImageKind.Face],  # 9
            [ImageKind.House],  # 10
            [ImageKind.Patterns],  # 11
            [ImageKind.Tool]],  # 12
        "eventrelatedold": [
            [ImageKind.Face],  # 5
            [ImageKind.House],  # 6
            [ImageKind.Tool]]  # 7

    }

    # reordered_indexes = np.array([0, 4,  7, 1, 5, 8, 2, 6, 9, 3])
    databases_to_check = ["eventrelatednatural", "blockdesign", "eventrelatedold"]
    # databases_to_check = ["blockdesign", "eventrelatedold"]
    xticks_labels = [
        [dataset_name_to_code[current_dataset_name] + "_" + get_contact_kinds_string(current_image_kind)
         for current_image_kind in modified_all_images_kinds[current_dataset_name]]
        for current_dataset_name in databases_to_check]

    x_props = [
        [(current_dataset_name, current_image_kind)
         for current_image_kind in modified_all_images_kinds[current_dataset_name]]
        for current_dataset_name in databases_to_check]

    xticks_labels = list(itertools.chain.from_iterable(xticks_labels))
    x_props = list(itertools.chain.from_iterable(x_props))


    statistics = {}
    for current_contact_kind in [ContactKind.All]:

        for current_modified_dataset, current_images_kind in x_props:

            for current_run_name, alignment_method in [("distances_alignment", lambda x, y: (x, y)),
                                                       ("spearman_alignment", lambda x, y: align_features_spearman_adapter(x, y)),
                                                       ("pearson_alignment", lambda x, y: align_features_pearson_adapter(x, y))]:


                test_group_size = 2 if current_modified_dataset == "everntrelatednatural" else 1

                try:
                    metadata = read_metadata(current_run_name + "/no_random_cross_validation",
                                             current_modified_dataset,
                                             current_contact_kind,
                                             current_images_kind,
                                             graph_configs[0].transformation_name,
                                             min_number_of_contacts)
                except:
                    a = 3

                number_of_indexes = metadata["number_of_indexes"]

                current_run_permutation = []

                low_low = 0
                low_high = 0
                high_high = 0
                total = 0

                for subject_1_index in range(number_of_indexes):
                    for subject_2_index in range(number_of_indexes):
                        if subject_1_index == subject_2_index:
                            continue

                        current_Data = read_data(current_run_name + "/no_random_cross_validation",
                                                 subject_1_index,
                                                 subject_2_index,
                                                 current_modified_dataset,
                                                 current_contact_kind,
                                                 current_images_kind,
                                                 graph_configs[0].transformation_name,
                                                 min_number_of_contacts,
                                                 test_group_size)

                        try:
                            subject_1_contact_types = current_Data["subject1_contacts_types"]
                            subject_2_contact_types = current_Data["subject2_contacts_types"]
                        except:
                            a = 3

                        for i, _ in enumerate(subject_1_contact_types):
                            total += 1
                            current_cotact_sub1 = subject_1_contact_types[i]
                            current_cotact_sub2 = subject_2_contact_types[i]

                            if current_cotact_sub1 == ContactKind.Low and current_cotact_sub2 == ContactKind.Low:
                                low_low += 1
                            elif current_cotact_sub1 == ContactKind.High and current_cotact_sub2 == ContactKind.High:
                                high_high += 1
                            else:
                                low_high += 1
                a = 3
                statistics[(current_modified_dataset, current_images_kind[0], current_run_name)] = {
                    "low_low": low_low,
                    "low_high": low_high,
                    "high_high": high_high,
                    "total": total
                }

    output_folder_path = "./output/graphs_new_11_14/spearman_measurement/matching_contacts.json"  # + curent_database_name + "/"

    with open(output_folder_path, "w") as f:
        json.dump(f, statistics, cls=SetEncoder)
    a = 3


output_folder_path = "./output/graphs_new_11_14/spearman_measurement/"  # + curent_database_name + "/"


class SetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return json.JSONEncoder.default(self, obj)


def get_min_max_ativation_values(subjects_data):

    min_value = 0
    max_value = -1000000000
    for subject in subjects_data:
        min_value = min(min_value, np.min(subject))
        max_value = max(max_value, np.max(subject))

    return (min_value, max_value)


def get_all_relevant_subject_data(brain_data, contact_kind_to_test):
    min_number_of_contacts = 4
    subjects_data = brain_data.get_brain_level_feature_image(contact_type=contact_kind_to_test,
                                                             make_feature_list=True,
                                                             use_people_with_high_and_low_contacts=False,
                                                             images_kinds=[ImageKind.All])
    subjects_data_indexes = [i for i, current_subject_data in
                             enumerate(subjects_data) if
                             current_subject_data.shape[1] >= min_number_of_contacts]

    subjects_data = [current_subject_data for i, current_subject_data in enumerate(subjects_data) if
                     i in subjects_data_indexes]

    return subjects_data

def create_bar_subject_activation_pattern(min_number_of_contacts):
    for base_alignment_name in  [
        # "spearman_alignment_normalize_images",
                                 "distances_alignment",
                                 "spearman_alignment",
                                 # "spearman_alignment_normalize_contacts",
                                 # "spearman_alignment_normalize_images"
                                ]:
        colors = ["red", "green", "blue", "cyan", "purple"]

        for database_name in databases_name:
            for current_contact_kind in contacts_to_test:
                brain_data = BrainData(database_name)
                subjects_data = get_all_relevant_subject_data(brain_data, current_contact_kind)
                min_max_values = get_min_max_ativation_values(subjects_data)

                for current_images_kind in all_images_kinds[database_name]:


                    metadata = read_metadata(base_alignment_name + "/no_random_cross_validation",
                                  database_name,
                                  current_contact_kind,
                                  current_images_kind,
                                  graph_configs[0].transformation_name,
                                  min_number_of_contacts)
                #

                    number_of_indexes = metadata["number_of_indexes"]
                    for subject_1_index in range(number_of_indexes):
                        for subject_2_index in range(number_of_indexes):
                            if subject_1_index >= subject_2_index:
                                continue


                            current_Data = read_data(base_alignment_name + "/no_random_cross_validation",
                                                     subject_1_index,
                                                     subject_2_index,
                                                     database_name,
                                                     current_contact_kind,
                                                     current_images_kind,
                                                     graph_configs[0].transformation_name,
                                                     min_number_of_contacts,
                                                     size_of_test_group=2)
                            feature_subject_1 = subjects_data[subject_1_index][:, current_Data['features_subject_1_indexes']]
                            feature_subject_2 = subjects_data[subject_2_index][:, current_Data['features_subject_2_indexes']]


                            # max_value = max(np.max(feature_subject_1), np.max(feature_subject_2))
                            # min_value = min(np.min(feature_subject_1), np.min(feature_subject_2))
                            bar_plot_ax_values = []
                            for i in range(feature_subject_1.shape[1]):
                                current_bar_plot_ax_row = []
                                plot_values_sub1 = []
                                plot_values_sub2 = []

                                for j, current_contact_imae_kind in enumerate(all_images_kinds[database_name]):
                                    plot_values_sub1.append(seaborn_utils.BarPlotCategoryValues(X=feature_subject_1[:, i][10 * j: 10 * (j + 1)],
                                                                                                 variance=None,
                                                                                                 label=current_contact_imae_kind[0].name,
                                                                                                 color=colors[j],
                                                                                                 base_ax_values=[10 * j + dd for dd in range(10)]
                                                                                                 ))

                                    plot_values_sub2.append(seaborn_utils.BarPlotCategoryValues(X=feature_subject_2[:, i][10 * j : 10 * (j + 1)],
                                                                                                 variance=None,
                                                                                                 label=current_contact_imae_kind[0].name,
                                                                                                 color=colors[j],
                                                                                                 base_ax_values=[10 * j + dd for dd in range(10)]
                                                                                                 ))


                                title_sub1 = "subject " + str(subject_1_index) if i == 0 else ""
                                current_bar_plot_ax_row.append(
                                    seaborn_utils.BarPlotAxValues(title=title_sub1,
                                                                  min_max = min_max_values,
                                                                  plot_values=plot_values_sub1,
                                                                  x_tick_labels=[str(dd) for dd in range(feature_subject_1.shape[0])],
                                                                  set_xticks=[dd for dd in range(feature_subject_1.shape[0])],
                                                                  should_show_legend=True)
                                )

                                title_sub2 = "subject " + str(subject_2_index) if i == 0 else ""

                                current_bar_plot_ax_row.append(
                                    seaborn_utils.BarPlotAxValues(title=title_sub2,
                                                                  min_max=min_max_values,
                                                                  plot_values=plot_values_sub2,
                                                                  x_tick_labels=[str(dd) for dd in range(feature_subject_2.shape[0])],
                                                                  set_xticks=[dd for dd in range(feature_subject_2.shape[0])],
                                                                  should_show_legend=True)
                                )
                                bar_plot_ax_values.append(current_bar_plot_ax_row)

                            folder_path = os.path.join(output_folder_path,
                                                     base_alignment_name,
                                                     str(min_number_of_contacts),
                                                     str(current_contact_kind),
                                                     "_".join([image_kind.name for image_kind in current_images_kind]),
                                                     "activaiton_bar_plots")


                            props = seaborn_utils.BarPlotProperties(
                                bar_plots= bar_plot_ax_values,
                                folder_path=folder_path,
                                file_name=str(subject_1_index) + "_" + str(subject_2_index) + ".jpeg",
                                title=base_alignment_name + "_" + str(subject_1_index) + "_" + str(subject_2_index)
                            )
                            seaborn_utils.output_bar(props)





#################### up to here#######


def create_bar_similarity_vs_test_group(min_number_of_contacts):
    for base_alignment_name in  ["spearman_alignment_normalize_images",
                                 "distances_alignment",
                                 "spearman_alignment",
                                 "spearman_alignment_normalize_contacts",
                                 "spearman_alignment_normalize_images"]:
        for database_name in databases_name:
            for current_contact_kind in contacts_to_test:
                for current_images_kind in all_images_kinds[database_name]:
                    #     return
                    all_results_test = defaultdict(lambda: defaultdict(float))
                    all_results_train = defaultdict(lambda: defaultdict(float))

                    should_return_permutation = True
                    for graph_config_index, graph_config in enumerate(graph_configs):
                        metadata = read_metadata(base_alignment_name + "/no_random_cross_validation",
                                      database_name,
                                      current_contact_kind,
                                      current_images_kind,
                                      graph_configs[0].transformation_name,
                                      min_number_of_contacts)

                        number_of_indexes = metadata["number_of_indexes"]
                        for test_group_size in range(1, 5, 1):
                            current_graph_results_test = []
                            current_graph_results_train = []
                            current_contact_kind_permutations_train = []
                            current_contact_kind_permutations_test = []

                            for subject_1_index in range(number_of_indexes):
                                current_subject_results_test = []
                                current_subject_results_train = []
                                current_subject_results_test_permutation = []
                                current_subject_results_train_permutation = []

                                for subject_2_index in range(number_of_indexes):
                                    #     if subject_1_index == subject_2_index:
                                    #         continue

                                    # #TODO: rempve (for tests)
                                    # if subject_1_index>= 2 or subject_2_index >=2:
                                    #     continue

                                    current_masks_correlations_train, \
                                    current_masks_correlations_test, \
                                    current_masks_correlations_permutations_train, \
                                    current_masks_correlations_permutations_test = get_pair_test_train_correlation(
                                        base_alignment_name,
                                        subject_1_index,
                                        subject_2_index,
                                        database_name,
                                        current_contact_kind,
                                        current_images_kind,
                                        graph_config,
                                        min_number_of_contacts,
                                        should_return_permutation=should_return_permutation,
                                        test_group_size=test_group_size)

                                    if should_return_permutation:
                                        current_subject_results_train_permutation.append(np.mean(current_masks_correlations_permutations_train))
                                        current_subject_results_test_permutation.append(np.mean(current_masks_correlations_permutations_test))

                                    current_subject_results_test.append(np.mean(np.array(current_masks_correlations_test)))
                                    current_subject_results_train.append(np.mean(np.array(current_masks_correlations_train)))

                                if should_return_permutation:
                                    current_contact_kind_permutations_test.append(current_subject_results_test_permutation)
                                    current_contact_kind_permutations_train.append(current_subject_results_train_permutation)
                                current_graph_results_test.append(current_subject_results_test)
                                current_graph_results_train.append(current_subject_results_train)

                            if should_return_permutation:
                                all_results_train["permutation"][test_group_size] = np.mean(current_contact_kind_permutations_train)
                                all_results_test["permutation"][test_group_size] = np.mean(current_contact_kind_permutations_test)
                            else:
                                a = 3

                            all_results_train[graph_config.transformation_name][test_group_size] = np.mean(current_graph_results_train)
                            all_results_test[graph_config.transformation_name][test_group_size] = np.mean(current_graph_results_test)

                        should_return_permutation = False

                    line_plot_results_test = []
                    line_plot_results_train = []

                    non_diagonal_mask = ~np.eye(len(current_graph_results_test), dtype=bool)
                    for (transformation_neuroscience_expression, current_transformation_name) in \
                            [("patterns", "permutation") ] + \
                            [(graph_config.neuroscienec_expression, graph_config.transformation_name) for graph_config in graph_configs]:

                        keys = sorted(list(all_results_test[current_transformation_name].keys()))

                        line_plot_results_test.append([transformation_neuroscience_expression,
                                                       keys,
                                                       [all_results_test[current_transformation_name][i] for i in keys]])

                        line_plot_results_train.append([transformation_neuroscience_expression,
                                                        keys,
                                                        [all_results_train[current_transformation_name][i] for i in range(1, 5, 1)]])

                    file_path = os.path.join(output_folder_path,
                                             base_alignment_name,
                                             str(min_number_of_contacts),
                                             str(current_contact_kind),
                                             "_".join([image_kind.name for image_kind in current_images_kind]),
                                             "line_plots",
                                             "lines_" + "_".join([image_kind.name for image_kind in current_images_kind]) + ".jpeg")

                    plot_correlates_transformation_lines(line_plot_results_test, line_plot_results_train, file_path, current_images_kind)




def plot_correlates_transformation_lines(line_plot_results_test, line_plot_results_train, file_path, current_images_kinds):
    fig_title = "_".join([current_images_kind_.name for current_images_kind_ in current_images_kinds])
    # current_graph_results_permutation = []
    current_graph_results_ordered_test = []
    current_graph_results_ordered_train = []
    colors = ["red", "green", "blue"]


    for i, current_line_plot_result_test in enumerate(line_plot_results_test):
        current_line_plot_result_train = line_plot_results_train[i]
        # current_graph_results_permutation.append(seaborn_utils.GraphIterationValues(np.arange(order_permutation.shape[0]),
        #                                                                             current_line_plot_result[1][order_permutation],
        #                                                                             line_title=current_line_plot_result[0],
        #                                                                             variances=None))
        current_graph_results_ordered_test.append(seaborn_utils.GraphIterationValues(current_line_plot_result_test[1],
                                                                                     current_line_plot_result_test[2],
                                                                                line_title=current_line_plot_result_test[0],
                                                                                variances=None,
                                                                                color=colors[i]))

        current_graph_results_ordered_train.append(seaborn_utils.GraphIterationValues(current_line_plot_result_train[1],
                                                                                      current_line_plot_result_train[2],
                                                                                      line_title=current_line_plot_result_train[0],
                                                                                      variances=None,
                                                                                      color=colors[i]))

    seaborn_utils.save_graph([
          seaborn_utils.OutputGraphPropertied(current_graph_results_ordered_test,
                                              x_label="test_group_size",
                                              y_label="spearman",
                                              title="test",
                                              file_path=file_path,
                                              max_y_lim=1.1,
                                              min_y_lim=-1.1,
                                              show_legend=True,
                                              x_labels=None),
        seaborn_utils.OutputGraphPropertied(current_graph_results_ordered_train,
                                            x_label="test_group_size",
                                            y_label="spearman",
                                            title="train",
                                            file_path=file_path,
                                            max_y_lim=1.1,
                                            min_y_lim=-1.1,
                                            show_legend=True,
                                            x_labels=None
                                            )
          ],
        fig_title)



###########


def create_bar_plot_check_contacts_similarity_cross_dataset(min_number_of_contacts):
    base_alignment_name = "coordinate_distances_alignment"
    modified_all_images_kinds = {
        "eventrelatednatural": [
            [ImageKind.Animals],  # 0
            [ImageKind.Patterns],  # 1
            [ImageKind.People],  # 2
            [ImageKind.Places],  # 3
            [ImageKind.Tools]],  # 4
        "blockdesign": [
            [ImageKind.Body],  # 8
            [ImageKind.Face],  # 9
            [ImageKind.House],  # 10
            [ImageKind.Patterns],  # 11
            [ImageKind.Tool]],  # 12
        "eventrelatedold": [
            [ImageKind.Face],  # 5
            [ImageKind.House],  # 6
            [ImageKind.Tool]]  # 7

    }

    # reordered_indexes = np.array([0, 4,  7, 1, 5, 8, 2, 6, 9, 3])
    reordered_indexes = np.array([2, 5, 9, 3, 6, 10, 4, 7, 12, 1, 11, 8, 0])

    xticks_labels = [
        [dataset_name_to_code[current_dataset_name] + "_" + get_contact_kinds_string(current_image_kind)
         for current_image_kind in modified_all_images_kinds[current_dataset_name]]
        for current_dataset_name in ["eventrelatednatural", "eventrelatedold", "blockdesign"]]

    x_props = [
        [(current_dataset_name, current_image_kind)
         for current_image_kind in modified_all_images_kinds[current_dataset_name]]
        for current_dataset_name in ["eventrelatednatural", "eventrelatedold", "blockdesign"]]

    xticks_labels = list(itertools.chain.from_iterable(xticks_labels))
    x_props = list(itertools.chain.from_iterable(x_props))

    xticks_labels = [xticks_labels[i].replace("People", "Face").replace("Places", "House") for i in
                     reordered_indexes] + ["Mean"]
    x_props = [x_props[i] for i in reordered_indexes]

    for current_contact_kind in contacts_to_test:
        values_all = defaultdict(list)
        variance_all = defaultdict(list)


        all_subjects_permutation = defaultdict(list)

        for current_modified_dataset, current_images_kind in x_props:
            values_for_z_test_train = []
            values_for_z_test_test = []

            for current_run_name, alignment_method in [("coordinate_distances_alignment", lambda x, y: (x, y)), ("spearman_alignment", lambda x, y: align_features_spearman_adapter(x, y))]:
                metadata = read_metadata(base_alignment_name + "/no_random_cross_validation",
                                         current_modified_dataset,
                                         current_contact_kind,
                                         current_images_kind,
                                         graph_configs[0].transformation_name,
                                         min_number_of_contacts)

                number_of_indexes = metadata["number_of_indexes"]

                current_run_permutation = []

                for subject_1_index in range(number_of_indexes):
                    for subject_2_index in range(number_of_indexes):
                        if subject_1_index == subject_2_index:
                            continue

                        permutation = get_pair_contacts_correlation(
                            base_alignment_name,
                            subject_1_index,
                            subject_2_index,
                            current_modified_dataset,
                            current_contact_kind,
                            current_images_kind,
                            graph_configs[0],
                            min_number_of_contacts,
                            alignment_method)

                        all_subjects_permutation[current_run_name].append(permutation)
                        current_run_permutation.append(permutation)

                values_all[current_run_name].append(np.mean(current_run_permutation))
                variance_all[current_run_name].append(sem(current_run_permutation))


            # significat_orthognoal_vs_linear_test.append(
            #     test_significant(x1=values_for_z_test_test[1], x2=values_for_z_test_test[0]))
            # significat_orthognoal_vs_linear_train.append(
            #     test_significant(x1=values_for_z_test_train[1], x2=values_for_z_test_train[0]))

        for current_run_name in ["coordinate_distances_alignment", "spearman_alignment"]:
            values_all[current_run_name].append(mean(all_subjects_permutation[current_run_name]))
            variance_all[current_run_name].append(sem(all_subjects_permutation[current_run_name]))
        #
        # significat_orthognoal_vs_linear_train.append(
        #     test_significant(x1=all_subjects_values_train["orthogonal"], x2=all_subjects_values_train["linear"]))
        # significat_orthognoal_vs_linear_test.append(
        #     test_significant(x1=all_subjects_values_test["orthogonal"], x2=all_subjects_values_test["linear"]))

        current_ax_bars = []
        # all_colors = ['red', 'green', "blue", "cyan", "purple"]
        all_colors = ["cyan", "purple"]

        base_ax_values = [0, 1, 2, 3.5, 4.5, 5.5, 7, 8, 9, 10.5, 11.5, 13, 14.5, 16]

        for i, current_run_name in enumerate(["coordinate_distances_alignment", "spearman_alignment"]):
            X_all = np.array(values_all[current_run_name])
            # all_x_test.append(X_test)
            current_base_ax_values = [base_ax_value + i * 0.25 for base_ax_value in base_ax_values]
            current_ax_bars.append(seaborn_utils.BarPlotCategoryValues(X=X_all,
                                                                            color=all_colors[i],
                                                                            variance=np.array(variance_all[current_run_name]),
                                                                            label=alignment_to_expression[current_run_name],
                                                                            base_ax_values=current_base_ax_values))

        # significat_orthognoal_vs_linear_test = [significat_orthognoal_vs_linear_test[i] for i in reordered_indexes]
        # significat_orthognoal_vs_linear_train = [significat_orthognoal_vs_linear_train[i] for i in reordered_indexes]

        # scatter_plot_test_x, scatter_plot_test_y = get_z_test_x_ticks(significat_orthognoal_vs_linear_test, all_x_test,
        #                                                               base_ax_values)
        #
        # scatter_plot_train_x, scatter_plot_train_y = get_z_test_x_ticks(significat_orthognoal_vs_linear_train,
        #                                                                 all_x_train, base_ax_values)

        bar_plot_ax_props = seaborn_utils.BarPlotAxValues("all values",
                                                          (0, 1.2),
                                                          current_ax_bars,
                                                          xticks_labels,
                                                          set_xticks=base_ax_values)

        folder_path = os.path.join(output_folder_path,
                                   base_alignment_name,
                                   str(min_number_of_contacts),
                                   str(current_contact_kind))

        seaborn_utils.output_bar(seaborn_utils.BarPlotProperties(
            bar_plots=[[bar_plot_ax_props]],
            folder_path=folder_path,
            file_name="spearman_vs_euclidean_contact_similarity",
            title="spearman_vs_euclidean_contact_similarity",
        )
            # marker_values=[
            #     [scatter_plot_test_x, scatter_plot_test_y],
            #     [scatter_plot_train_x, scatter_plot_train_y]
            # ]
        )
        a = 3


def align_features_spearman_adapter(features_1, features_2):
    features_subject_1_aligned, features_subject_2_aligned, _, _ = align_features_spearman(features_1, features_2, None, None)
    return features_subject_1_aligned, features_subject_2_aligned

def align_features_pearson_adapter(features_1, features_2):
    features_subject_1_aligned, features_subject_2_aligned, _, _ = align_features_pearson(features_1, features_2, None, None)
    return features_subject_1_aligned, features_subject_2_aligned


def create_bar_plot_cross_dataset(min_number_of_contacts):
    base_alignment_name = "coordinate_distances_alignment"
    modified_all_images_kinds = {
        "eventrelatednatural": [
            [ImageKind.Animals],  # 0
            [ImageKind.Patterns],  # 1
            [ImageKind.People],  # 2
            [ImageKind.Places],  # 3
            [ImageKind.Tools]],  # 4
        "blockdesign": [
            [ImageKind.Body],  # 8
            [ImageKind.Face],  # 9
            [ImageKind.House],  # 10
            [ImageKind.Patterns],  # 11
            [ImageKind.Tool]],  # 12
        "eventrelatedold": [
            [ImageKind.Face],  # 5
            [ImageKind.House],  # 6
            [ImageKind.Tool]]  # 7

    }

    # reordered_indexes = np.array([2, 5, 9, 3, 6, 10, 4, 7, 12, 1, 11, 8, 0])
    reordered_indexes = np.array([2, 5, 9, 3, 6, 10, 4, 7, 12, 1, 11, 8, 0])

    xticks_labels = [
        [dataset_name_to_code[current_dataset_name] + "_" + get_contact_kinds_string(current_image_kind)
         for current_image_kind in modified_all_images_kinds[current_dataset_name]]
        for current_dataset_name in ["eventrelatednatural"]]
        # for current_dataset_name in ["eventrelatednatural", "eventrelatedold", "blockdesign"]]

    x_props = [
        [(current_dataset_name, current_image_kind)
         for current_image_kind in modified_all_images_kinds[current_dataset_name]]
        for current_dataset_name in ["eventrelatednatural"]]
        # for current_dataset_name in ["eventrelatednatural", "eventrelatedold", "blockdesign"]]

    xticks_labels = list(itertools.chain.from_iterable(xticks_labels))
    x_props = list(itertools.chain.from_iterable(x_props))

    xticks_labels = [xticks_labels[i].replace("People", "Face").replace("Places", "House") for i in
                     reordered_indexes] + ["Mean"]
    # x_props = [x_props[i] for i in reordered_indexes]

    for current_contact_kind in contacts_to_test:
        values_all = defaultdict(list)
        variance_all = defaultdict(list)

        significat_orthognoal_vs_linear_test = []
        significat_orthognoal_vs_linear_train = []

        # for current_modified_dataset in ["eventrelatednatural", "blockdesign", "eventrelatedold"]:
        #     current_dataset_modified_images_kinds = modified_all_images_kinds[current_modified_dataset]
        #     for current_images_kind in current_dataset_modified_images_kinds:
        #         pass
        all_subjects_permutation = defaultdict(list)

        for current_modified_dataset, current_images_kind in x_props:
            values_for_z_test_train = []
            values_for_z_test_test = []
            alignments_props = [("coordinate_distances_alignment", lambda x,y : (x, y)),
                                ("spearman_alignment", lambda x,y: align_features_spearman_adapter(x,y))]
            for current_run_name, alignment_method in []:
                metadata = read_metadata(base_alignment_name + "/no_random_cross_validation",
                                         current_modified_dataset,
                                         current_contact_kind,
                                         current_images_kind,
                                         graph_configs[0].transformation_name,
                                         min_number_of_contacts)

                number_of_indexes = metadata["number_of_indexes"]

                current_run_permutation = []

                for subject_1_index in range(number_of_indexes):
                    for subject_2_index in range(number_of_indexes):
                        if subject_1_index == subject_2_index:
                            continue

                        permutation = get_pair_correlations(
                            base_alignment_name,
                            subject_1_index,
                            subject_2_index,
                            current_modified_dataset,
                            current_contact_kind,
                            current_images_kind,
                            graph_configs[0],
                            min_number_of_contacts,
                            alignment_method,
                            test_group_size=2)

                        all_subjects_permutation[current_run_name].append(permutation)
                        current_run_permutation.append(permutation)

                values_all[current_run_name].append(np.mean(current_run_permutation))
                variance_all[current_run_name].append(sem(current_run_permutation))


            # significat_orthognoal_vs_linear_test.append(
            #     test_significant(x1=values_for_z_test_test[1], x2=values_for_z_test_test[0]))
            # significat_orthognoal_vs_linear_train.append(
            #     test_significant(x1=values_for_z_test_train[1], x2=values_for_z_test_train[0]))

        for current_run_name in ["coordinate_distances_alignment", "spearman_alignment"]:
            values_all[current_run_name].append(mean(all_subjects_permutation[current_run_name]))
            variance_all[current_run_name].append(sem(all_subjects_permutation[current_run_name]))
        #
        # significat_orthognoal_vs_linear_train.append(
        #     test_significant(x1=all_subjects_values_train["orthogonal"], x2=all_subjects_values_train["linear"]))
        # significat_orthognoal_vs_linear_test.append(
        #     test_significant(x1=all_subjects_values_test["orthogonal"], x2=all_subjects_values_test["linear"]))

        current_ax_bars = []
        # all_colors = ['red', 'green', "blue", "cyan", "purple"]
        all_colors = ["cyan", "purple"]

        base_ax_values = [0, 1, 2, 3.5, 4.5, 5.5, 7, 8, 9, 10.5, 11.5, 13, 14.5, 16]

        for i, current_run_name in enumerate(["coordinate_distances_alignment", "spearman_alignment"]):
            X_all = np.array(values_all[current_run_name])
            # all_x_test.append(X_test)
            current_base_ax_values = [base_ax_value + i * 0.25 for base_ax_value in base_ax_values]
            current_ax_bars.append(seaborn_utils.BarPlotCategoryValues(X=X_all,
                                                                            color=all_colors[i],
                                                                            variance=np.array(variance_all[current_run_name]),
                                                                            label=alignment_to_expression[current_run_name],
                                                                            base_ax_values=current_base_ax_values))

        # significat_orthognoal_vs_linear_test = [significat_orthognoal_vs_linear_test[i] for i in reordered_indexes]
        # significat_orthognoal_vs_linear_train = [significat_orthognoal_vs_linear_train[i] for i in reordered_indexes]

        # scatter_plot_test_x, scatter_plot_test_y = get_z_test_x_ticks(significat_orthognoal_vs_linear_test, all_x_test,
        #                                                               base_ax_values)
        #
        # scatter_plot_train_x, scatter_plot_train_y = get_z_test_x_ticks(significat_orthognoal_vs_linear_train,
        #                                                                 all_x_train, base_ax_values)

        bar_plot_ax_props = seaborn_utils.BarPlotAxValues("all values", (0, 1.2), current_ax_bars, xticks_labels,
                                                                set_xticks=base_ax_values)

        folder_path = os.path.join(output_folder_path,
                                   base_alignment_name,
                                   str(min_number_of_contacts),
                                   str(current_contact_kind))

        seaborn_utils.output_bar(seaborn_utils.BarPlotProperties(
            bar_plots=[[bar_plot_ax_props]],
            folder_path=folder_path,
            file_name="spearman_vs_euclidean_permutation_similarity",
            title="spearman_vs_euclidean_permutation_similarity",
        )
            # marker_values=[
            #     [scatter_plot_test_x, scatter_plot_test_y],
            #     [scatter_plot_train_x, scatter_plot_train_y]
            # ]
        )
        a = 3


def test_significant(x1, x2):
    if len(x1) > 30:
        return ztest(x1=x1, x2=x2)
    return ttest_ind(x1, x2)
    pass


def get_z_test_x_ticks(all_z_values, X_values, base_ax_values):
    result_x = [[], [], []]
    result_y = [[], [], []]

    for i, z_value in enumerate(all_z_values):
        x_value = max([X_values[j][i] for j in range(len(X_values))]) + 0.1
        try:
            if z_value[1] < 0.05:
                result_x[0].append(base_ax_values[i])
                result_y[0].append(x_value)
        except:
            a = 3
        if z_value[1] < 0.01:
            result_x[1].append(0.05 + base_ax_values[i])
            result_y[1].append(x_value)

        if z_value[1] < 0.001:
            result_x[2].append(0.1 + base_ax_values[i])
            result_y[2].append(x_value)

    # return z_values[0] < -1.96 and z_values[1] < 0.05
    return result_x, result_y


def get_pair_train_test_correlations(feature_subject_1,
                                     feature_subject_2,
                                     current_transformation_matrix,
                                     current_masks,
                                     should_return_permutation):
    # current_transformation_matrix = transformation_matrices[i]
    # current_masks = masks[i]
    current_train_mask = np.array(current_masks.train_mask, dtype=bool)
    current_test_mask = np.array(current_masks.test_mask, dtype=bool)

    current_pair_permutation_correlation_train = None
    current_pair_permutation_correlation_test = None

    if should_return_permutation:
        current_pair_permutation_correlation_train = get_correlation_value(feature_subject_1,
                                                                           feature_subject_2,
                                                                           mask=current_train_mask)
        current_pair_permutation_correlation_test = get_correlation_value(feature_subject_1,
                                                                          feature_subject_2,
                                                                          mask=current_test_mask)

    current_pair_train_correlations = get_correlation_value(feature_subject_1,
                                                            feature_subject_2,
                                                            current_transformation_matrix,
                                                            mask=current_train_mask)

    current_pair_test_correlations = get_correlation_value(feature_subject_1,
                                                           feature_subject_2,
                                                           current_transformation_matrix,
                                                           mask=current_test_mask)
    return current_pair_train_correlations, \
           current_pair_test_correlations, \
           current_pair_permutation_correlation_train, \
           current_pair_permutation_correlation_test


def get_images_pair_to_pair_values(modified_images_kinds, results, number_of_indexes, information_name):
    results_values_mean = []
    result_values_sem = []
    for image_kind_1 in modified_images_kinds:
        current_results_values_mean = []
        current_results_values_sem = []
        for image_kind_2 in modified_images_kinds:
            if image_kind_1 == image_kind_2:
                new_images_kinds = image_kind_1
            else:
                new_images_kinds = image_kind_1 + image_kind_2

            if (image_kind_1 == [ImageKind.All] and image_kind_2 != [ImageKind.All]) or \
                    (image_kind_1 != [ImageKind.All] and image_kind_2 == [ImageKind.All]):
                continue
            mean_value, sem_value = get_pair_image_kind_value(results, number_of_indexes, new_images_kinds,
                                                              information_name)
            current_results_values_mean.append(mean_value)
            current_results_values_sem.append(sem_value)
        results_values_mean.append(np.array(current_results_values_mean))
        result_values_sem.append(np.array(current_results_values_sem))
    return np.array(results_values_mean), np.array(result_values_sem)


def get_pair_image_kind_value(result, number_of_indexes, image_kind, information_name):
    values = [result[tuple(image_kind)][subject_1_index][subject_2_index][information_name]
              for subject_1_index in range(number_of_indexes)
              for subject_2_index in range(number_of_indexes - 1)]
    return mean(values), sem(values)


def create_pair_images_kinds_matrix(current_run_name,
                                    min_number_of_contacts,
                                    database_name):
    modified_graph_configs: List[GraphConfig] = [
        GraphConfig(transformation_name="orthogonal", neuroscienec_expression="relations"),
        GraphConfig(transformation_name="linear", neuroscienec_expression="topology")
    ]

    modified_images_kinds = deepcopy(images_kinds)
    # modified_images_kinds.remove([ImageKind.All])
    # modified_images_kinds = deepcopy(modified_all_images_kinds)

    configs_images_to_test = []
    for image_kind_1_index, image_kind_1 in enumerate(modified_images_kinds):
        configs_images_to_test.append(image_kind_1)

    for image_kind_1_index, image_kind_1 in enumerate(modified_images_kinds):
        for image_kind_2_index, image_kind_2 in enumerate(modified_images_kinds):
            if image_kind_1_index > image_kind_2_index or ImageKind.All in image_kind_2 or ImageKind.All in image_kind_1 or image_kind_2 == image_kind_1:
                continue
            configs_images_to_test.append(image_kind_1 + image_kind_2)

    for image_kind_1_index, image_kind_1 in enumerate(modified_images_kinds):
        for image_kind_2_index, image_kind_2 in enumerate(modified_images_kinds):
            for image_kind_3_index, image_kind_3 in enumerate(modified_images_kinds):
                if image_kind_1_index > image_kind_2_index or image_kind_2_index > image_kind_3_index or image_kind_1_index > image_kind_3_index:
                    continue
                if ImageKind.All in image_kind_2 or ImageKind.All in image_kind_1 or ImageKind.All in image_kind_3:
                    continue

                if image_kind_1 != image_kind_2 and image_kind_1 != image_kind_3 and image_kind_3 != image_kind_2:
                    configs_images_to_test.append(image_kind_1 + image_kind_2 + image_kind_3)

    for current_contact_kind in contacts_to_test:
        result = [defaultdict(list) for _ in modified_graph_configs]
        for new_images_kinds in configs_images_to_test:
            for current_graph_index, current_graph_config in enumerate(modified_graph_configs):
                metadata = read_metadata(current_run_name,
                                         database_name,
                                         current_contact_kind,
                                         new_images_kinds,
                                         graph_configs[0].transformation_name,
                                         min_number_of_contacts)

                number_of_indexes = metadata["number_of_indexes"]

                # TODO: remove
                # number_of_indexes = 3
                a = 3

                for subject_1_index in range(number_of_indexes):
                    result[current_graph_index][tuple(new_images_kinds)].append([])
                    for subject_2_index in range(number_of_indexes):
                        if subject_1_index == subject_2_index:
                            continue

                        # # TODO: remove
                        # if subject_1_index >= 2 or subject_2_index >= 2:
                        #     continue

                        pair_train_values, pair_test_values, _, _ = get_pair_correlations(current_run_name,
                                                                                          subject_1_index,
                                                                                          subject_2_index,
                                                                                          database_name,
                                                                                          current_contact_kind,
                                                                                          new_images_kinds,
                                                                                          current_graph_config,
                                                                                          min_number_of_contacts)
                        result[current_graph_index][tuple(new_images_kinds)][subject_1_index].append({
                            "train_correlation": np.mean(pair_train_values),
                            "train_variance": sem(pair_train_values),
                            "test_correlation": np.mean(pair_test_values),
                            "test_variance": sem(pair_test_values)
                        })

        results_ratio = []
        results_sem = []
        for current_config_images_index, current_config_images in enumerate(configs_images_to_test):
            current_images_kinds_result_ratio = []
            current_images_kinds_result_sem = []
            for current_graph_index, current_graph_config in enumerate(modified_graph_configs):
                metadata = read_metadata(current_run_name,
                                         database_name,
                                         current_contact_kind,
                                         current_config_images,
                                         graph_configs[0].transformation_name,
                                         min_number_of_contacts)
                number_of_indexes = metadata["number_of_indexes"]
                # number_of_indexes = 3 #TODO: remove

                mean_value, sem_value = get_pair_image_kind_value(result[current_graph_index], number_of_indexes,
                                                                  current_config_images, "test_correlation")
                current_images_kinds_result_ratio.append(mean_value)
                current_images_kinds_result_sem.append(sem_value)
            results_ratio.append(current_images_kinds_result_ratio[0] / current_images_kinds_result_ratio[1])
            results_sem.append(current_images_kinds_result_sem[0] + current_images_kinds_result_sem[1])

        max_ratio = (np.array(results_ratio) + np.array(results_sem)).max() * 1.1
        min_ratio = (np.array(results_ratio) - np.array(results_sem)).min() * 0.9

        x_labels = ["_".join([current_images_kind.name for current_images_kind in current_config_images_kind]) for
                    current_config_images_kind in configs_images_to_test]
        title_relation_graph = "_".join(
            [current_grpah_config.neuroscienec_expression for current_grpah_config in modified_graph_configs])

        ordered_indexes = list(reversed(list(np.argsort(np.array(results_ratio)))))

        x_labels = [x_labels[i] for i in ordered_indexes]
        results_sem = [results_sem[i] for i in ordered_indexes]
        results_ratio = [results_ratio[i] for i in ordered_indexes]

        folder_path = os.path.join(output_folder_path,
                                   current_run_name,
                                   str(min_number_of_contacts),
                                   str(current_contact_kind))

        bar_plot_ax_values = [[
            get_bar_plot_pair_categories_dataset(results_ratio,
                                                 "Purity",
                                                 x_labels,
                                                 results_sem,
                                                 (min_ratio, max_ratio))
        ]]
        seaborn_utils.output_bar(
            seaborn_utils.BarPlotProperties(bar_plot_ax_values, folder_path + "/pair_images_type/bar", "purity", ""))
        a = 3


def get_bar_plot_pair_categories_dataset(results_ratio,
                                         title_relation_graph,
                                         x_labels,
                                         results_sem,
                                         min_max_values
                                         ):
    colors = {
        "Al": "black",  # All
        "An": "red",  # Animals
        "Bo": "blue",  # Body
        "Fa": "cyan",  # Face
        "Ho": "brown",  # House
        "Pa": "orange",  # Patterns
        "Pe": "purple",  # People
        "Pl": "pink",  # Places
        "To": "gold",  # Tooxl + Tools

    }
    plot_values = []
    x_labels = [[current_label[:2] for current_label in x_label.replace("People", "Faces").split("_")] for x_label in
                x_labels]

    for current_label_index in range(len(x_labels)):
        current_labels = x_labels[current_label_index]
        each_label_part = (results_ratio[current_label_index] - min_max_values[0]) / len(current_labels)
        bottom = 0
        for current_labels_index, current_label in enumerate(current_labels):
            if current_labels_index == len(current_labels) - 1:
                variance = np.array(results_sem[current_label_index])
            else:
                variance = 0
            color = colors[current_label]
            current_label_part = each_label_part
            if current_labels_index == 0:
                current_label_part += min_max_values[0]
            plot_values.append(seaborn_utils.BarPlotCategoryValues([current_label_part],
                                                                   color,
                                                                   label="",
                                                                   variance=variance,
                                                                   width=0.5,
                                                                   base_ax_values=[current_label_index],
                                                                   bottom=[bottom]))

            bottom += current_label_part

    x_labels = ["+".join(current_labels) for current_labels in x_labels]

    return seaborn_utils.BarPlotAxValues(title_relation_graph,
                                         min_max_values,
                                         y_label="orthogonal to linear ratio on test",
                                         x_tick_labels=x_labels,
                                         plot_values=plot_values,
                                         dashed_horizontal_line=1,
                                         set_xticks=list(range(len(x_labels))))


def get_pair_correlations(current_run_name,
                          subject_1_index,
                          subject_2_index,
                          database_name,
                          current_contact_kind,
                          new_images_kinds,
                          current_graph_config,
                          min_number_of_contacts,
                          alignment_method,
                          test_group_size):
    current_Data = read_data(current_run_name + "/no_random_cross_validation",
                             subject_1_index,
                             subject_2_index,
                             database_name,
                             current_contact_kind,
                             new_images_kinds,
                             current_graph_config.transformation_name,
                             min_number_of_contacts,
                             test_group_size)

    feature_subject_1 = current_Data["features_subject_1"]
    feature_subject_2 = current_Data["features_subject_2"]
    if alignment_method != None:
        feature_subject_1, feature_subject_2 = alignment_method(feature_subject_1, feature_subject_2)
    return get_correlation_value(feature_subject_1,
                                 feature_subject_2)





def get_pair_contacts_correlation(current_run_name,
                                  subject_1_index,
                                  subject_2_index,
                                  database_name,
                                  current_contact_kind,
                                  new_images_kinds,
                                  current_graph_config,
                                  min_number_of_contacts,
                                  alignment_method,
                                  test_group_size):
    current_Data = read_data(current_run_name + "/no_random_cross_validation",
                             subject_1_index,
                             subject_2_index,
                             database_name,
                             current_contact_kind,
                             new_images_kinds,
                             current_graph_config.transformation_name,
                             min_number_of_contacts,
                             test_group_size)

    feature_subject_1 = current_Data["features_subject_1"]
    feature_subject_2 = current_Data["features_subject_2"]

    feature_subject_1, feature_subject_2 = alignment_method(feature_subject_1, feature_subject_2)

    return get_correlation_value(feature_subject_1.T,
                                 feature_subject_2.T)

def get_pair_test_train_correlation(current_run_name,
                                  subject_1_index,
                                  subject_2_index,
                                  database_name,
                                  current_contact_kind,
                                  new_images_kinds,
                                  current_graph_config,
                                  min_number_of_contacts,
                                  should_return_permutation,
                                  test_group_size):
    current_Data = read_data(current_run_name + "/no_random_cross_validation",
                             subject_1_index,
                             subject_2_index,
                             database_name,
                             current_contact_kind,
                             new_images_kinds,
                             current_graph_config.transformation_name,
                             min_number_of_contacts,
                             test_group_size)

    feature_subject_1 = current_Data["features_subject_1"]
    feature_subject_2 = current_Data["features_subject_2"]

    transformation_matrices = current_Data["transformation_matrices"]
    masks = current_Data["masks"]
    current_pair_test_correlations = []
    current_pair_train_correlations = []
    if should_return_permutation:
        current_pair_test_permutations = []
        current_pair_train_permutations = []

    for i in range(len(transformation_matrices)):
        # TODO: remove
        # if i > 1:
        #     continue
        current_transformation_matrix = transformation_matrices[i]
        current_masks = masks[i]
        train_correlaiton, test_correlation, permutation_train_correlation, permutation_test_correlation = \
            get_pair_train_test_correlations(feature_subject_1,
                                             feature_subject_2,
                                             current_transformation_matrix,
                                             current_masks,
                                             should_return_permutation)
        if should_return_permutation:
            current_pair_test_permutations.append(permutation_test_correlation)
            current_pair_train_permutations.append(permutation_train_correlation)

        current_pair_train_correlations.append(train_correlaiton)
        current_pair_test_correlations.append(test_correlation)
    if should_return_permutation:
        return current_pair_train_correlations, current_pair_test_correlations, current_pair_train_permutations, current_pair_test_permutations
    return current_pair_train_correlations, current_pair_test_correlations, None, None


def get_masked_features_correlation(features_1, features_2, mask, transformation_matrix=None, use_distances=False):
    feature_subject_1_masked = features_1[mask].reshape(-1, mask.shape[1])
    feature_subject_2_masked = features_2[mask].reshape(-1, mask.shape[1])
    if transformation_matrix is None:
        return get_correlation_value(feature_subject_1_masked,
                                     feature_subject_2_masked,
                                     use_distances=use_distances)

    return get_correlation_value(feature_subject_1_masked,
                                 feature_subject_2_masked,
                                 transformation_matrix=transformation_matrix,
                                 use_distances=use_distances)


def get_contact_kinds_string(current_images_kinds):
    return "_".join([image_kind.name for image_kind in current_images_kinds])




graph_configs: List[GraphConfig] = [
    # GraphConfig(transformation_name="orthogonal_normalize"),
    GraphConfig(transformation_name="orthogonal", neuroscienec_expression="relations"),
    GraphConfig(transformation_name="linear", neuroscienec_expression="affine")
    # GraphConfig(transformation_name="linear_normalized")
]


def get_correlation_value(first_subject_features, second_subject_features, transformation_matrix=None, mask=None, use_distances=False):
    if mask is not None:
        first_subject_features = first_subject_features[mask].reshape(-1, first_subject_features.shape[1])
        second_subject_features = second_subject_features[mask].reshape(*first_subject_features.shape)

    if use_distances:
        return get_distance_corelation_value(first_subject_features,
                                             second_subject_features,
                                             transformation_matrix=transformation_matrix)
    return get_spearman_corelation_value(first_subject_features,
                                         second_subject_features,
                                         transformation_matrix=transformation_matrix)


def get_distance_corelation_value(first_subject_features, second_subject_features, transformation_matrix=None):
    if transformation_matrix is not None:
        return np.mean(np.diag(cdist(first_subject_features @ transformation_matrix, second_subject_features)))
    return np.mean(np.diag(cdist(first_subject_features, second_subject_features)))


def get_spearman_corelation_value(first_subject_features, second_subject_features, transformation_matrix=None):
    if transformation_matrix is not None:
        return np.mean(np.diag(numpy_utils.spearman_matrix(first_subject_features @ transformation_matrix, second_subject_features)))
    return np.mean(np.diag(numpy_utils.spearman_matrix(first_subject_features, second_subject_features)))


databases_name = ["eventrelatednatural", "blockdesign", "eventrelatedold"]
# databases_name = ["blockdesign", "eventrelatedold"]

databases_name = ["eventrelatednatural"]
# databases_name = ["blockdesign"]
# databases_name = ["eventrelatedold"]
contacts_to_test = [ContactKind.High, ContactKind.All]
# contacts_to_test = [ContactKind.LowAndFaceSelective, ContactKind.FaceSelective]
contacts_to_test = [ContactKind.High]



output_folder_path = "./output/graphs_new_09_15/"  # + curent_database_name + "/"
starting_time = datetime.now()
create_graph_files_all_kinds()
print("finish running " + str(datetime.now() - starting_time))





