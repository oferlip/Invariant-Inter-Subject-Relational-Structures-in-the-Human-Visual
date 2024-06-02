from typing import List, Tuple
import os
import numpy as np
from scipy.spatial.distance import cdist
from dataclasses import dataclass
from multiprocessing.pool import ThreadPool
import itertools
import sys

absolute_path_to_repository = "Insert path to repository"

sys.path.append(os.path.dirname(f"{absolute_path_to_repository}/code/similarity/"))
sys.path.append(os.path.dirname(f"{absolute_path_to_repository}/code/"))
sys.path.append(os.path.dirname(absolute_path_to_repository))


from similarity.utils import seaborn_utils


from statistics import mean
from datetime import datetime
from collections import defaultdict
import itertools
from statsmodels.stats.weightstats import ztest as ztest
from scipy.stats import sem, pearsonr
from matplotlib.patches import Patch
from scipy.stats import wilcoxon


transformation_name_to_expression={
    "permutation": "Pattern Coding",
    "orthogonal": "Relational coding",
    "linear": "Linear coding"
}

dataset_name_to_code = {
    "eventrelatednatural": "set1",
    "eventrelatedold": "set2",
    "blockdesign" : "set3"
}

from similarity.data_stucture.BrainData import ContactKind, ImageKind, BrainData
from similarity.results_code.result_cache_utils import read_data, read_metadata
from similarity.utils import numpy_utils




@dataclass
class GraphConfig:
    transformation_name: str
    neuroscienec_expression: str

all_images_kinds = {
    "eventrelatednatural": [
                            [ImageKind.All],
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


def create_graph_files_all_kinds(current_run_name: str,
                                 graph_configs: List[GraphConfig],
                                 contact_kinds_to_test: List[ContactKind],
                                 database_name):



    for current_images_kinds in images_kinds:
        create_lins(current_images_kinds,
                    current_run_name,
                    min_number_of_contacts,
                    database_name)


def create_cross_dataset_bar_plots(current_run_name,
                                  min_number_of_contacts):
    
    modified_all_images_kinds = {
        "eventrelatednatural": [
            [ImageKind.Animals], #0
            [ImageKind.Patterns],# 1
            [ImageKind.People], #2
            [ImageKind.Places], #3
            [ImageKind.Tools]], #4
        "blockdesign": [
            [ImageKind.Body],# 8
            [ImageKind.Face], # 9
            [ImageKind.House],# 10
            [ImageKind.Patterns], # 11
            [ImageKind.Tool]], # 12
        "eventrelatedold": [
            [ImageKind.Face],# 5
            [ImageKind.House],# 6
            [ImageKind.Tool]] #7


        }

    reordered_indexes = np.array([2, 5, 9, 3, 6, 10, 4, 12, 1, 11, 8, 0]) # WHEN SET2 TOOLS IS out

    xticks_labels = [
        [dataset_name_to_code[current_dataset_name] + "_" + get_contact_kinds_string(current_image_kind)
         for current_image_kind in modified_all_images_kinds[current_dataset_name]]
        for current_dataset_name in ["eventrelatednatural", "eventrelatedold", "blockdesign"]]

    x_props = [
        [(current_dataset_name , current_image_kind)
         for current_image_kind in modified_all_images_kinds[current_dataset_name]]
        for current_dataset_name in ["eventrelatednatural", "eventrelatedold", "blockdesign"]]


    xticks_labels = list(itertools.chain.from_iterable(xticks_labels))
    x_props = list(itertools.chain.from_iterable(x_props))


    xticks_labels = [xticks_labels[i].replace("People", "Face").replace("Places", "House") for i in reordered_indexes] #+ ["Mean"]
    x_props = [x_props[i] for i in reordered_indexes]

    time_cross_dataset_current = datetime.now()
    for current_contact_kind in contacts_to_test:
        values_test = defaultdict(list)
        variance_test = defaultdict(list)
        values_train = defaultdict(list)
        variance_train = defaultdict(list)

        significat_orthognoal_vs_linear_test = []
        significat_orthognoal_vs_linear_train = []


        all_subjects_values_train = defaultdict(list)
        all_subjects_values_test = defaultdict(list)

        for current_modified_dataset, current_images_kind in x_props:

            print("time before last :", datetime.now() - time_cross_dataset_current)
            time_cross_dataset_current = datetime.now()
            print("current_modified_dataset:", current_modified_dataset, ", current_images_kind: ", current_images_kind, ", current_contact_kind:", current_contact_kind)

            should_calc_permutation = True
            values_for_z_test_train = []
            values_for_z_test_test = []
            for current_graph_index, current_graph_config in enumerate(graph_configs):

                metadata = read_metadata(current_run_name,
                                         current_modified_dataset,
                                         current_contact_kind,
                                         current_images_kind,
                                         graph_configs[0].transformation_name,
                                         min_number_of_contacts)
                number_of_indexes = metadata["number_of_indexes"]

                if should_calc_permutation:
                    current_graph_permutation_test = []
                    current_graph_permutation_train = []

                current_graph_correlation_test = []
                current_graph_correlation_train = []

                for subject_1_index in range(number_of_indexes):
                    for subject_2_index in range(number_of_indexes):
                        if subject_1_index == subject_2_index:
                            continue

                        pair_train_values, pair_test_values, permutation_train, permutation_test = get_pair_correlations(current_run_name,
                                                                                                                         subject_1_index,
                                                                                                                         subject_2_index,
                                                                                                                         current_modified_dataset,
                                                                                                                         current_contact_kind,
                                                                                                                         current_images_kind,
                                                                                                                         current_graph_config,
                                                                                                                         min_number_of_contacts,
                                                                                                                         should_calc_permutation)
                        if should_calc_permutation:

                            all_subjects_values_train["permutation"].append(np.mean(permutation_train))
                            all_subjects_values_test["permutation"].append(np.mean(permutation_test))

                            current_graph_permutation_train.append(np.mean(permutation_train))
                            current_graph_permutation_test.append(np.mean(permutation_test))

                        current_graph_correlation_train.append(np.mean(pair_train_values))
                        current_graph_correlation_test.append(np.mean(pair_test_values))

                        all_subjects_values_train[current_graph_config.transformation_name].append(np.mean(pair_train_values))
                        all_subjects_values_test[current_graph_config.transformation_name].append(np.mean(pair_test_values))


                if should_calc_permutation:
                    values_test["permutation"].append(np.mean(current_graph_permutation_test))
                    values_train["permutation"].append(np.mean(current_graph_permutation_train))

                    variance_test["permutation"].append(sem(current_graph_permutation_test))
                    variance_train["permutation"].append(sem(current_graph_permutation_train))

                values_for_z_test_train.append(current_graph_correlation_train)
                values_for_z_test_test.append(current_graph_correlation_test)
                values_test[current_graph_config.transformation_name].append(np.mean(current_graph_correlation_test))
                values_train[current_graph_config.transformation_name].append(np.mean(current_graph_correlation_train))

                variance_test[current_graph_config.transformation_name].append(sem(current_graph_correlation_test))
                variance_train[current_graph_config.transformation_name].append(sem(current_graph_correlation_train))

                should_calc_permutation = False
            significat_orthognoal_vs_linear_test.append(test_significant(x1=current_graph_permutation_test, x2=values_for_z_test_test[0]))
            # significat_orthognoal_vs_linear_test.append(test_significant(x1=values_for_z_test_test[0], x2=values_for_z_test_test[1]))


        current_ax_bars_test = []
        current_ax_bars_train = []
        all_colors = [ 'red', 'green', "blue", "cyan"]

        base_ax_values = [0, 1,  2, 4.5, 5.5, 6.5, 9, 10, 12.5, 13.5, 16, 18.5] #without set 2 tools
        all_x_test = []
        all_x_train = []


        # overall mean
        for i, current_plot_name in enumerate(["permutation", "orthogonal", "linear"]):
            X_test = np.array(values_test[current_plot_name])
            all_x_test.append(X_test)
            current_base_ax_values = [base_ax_value + i * 0.25 for base_ax_value in base_ax_values]
            current_ax_bars_test.append(seaborn_utils.BarPlotCategoryValues(X=X_test,
                                                                       color=all_colors[i],
                                                                       variance=np.array(variance_test[current_plot_name]),
                                                                       label=transformation_name_to_expression[current_plot_name],
                                                                       base_ax_values=current_base_ax_values))
            X_train = np.array(values_train[current_plot_name])
            all_x_train.append(X_train)
            current_ax_bars_train.append(seaborn_utils.BarPlotCategoryValues(X=X_train,
                                                                            color=all_colors[i],
                                                                            variance=np.array(variance_train[current_plot_name]),
                                                                            label=transformation_name_to_expression[current_plot_name],
                                                                            base_ax_values=current_base_ax_values))


#######
        

        scatter_plot_test_x, scatter_plot_test_y = get_z_test_x_ticks(significat_orthognoal_vs_linear_test, all_x_test, base_ax_values)

        scatter_plot_train_x, scatter_plot_train_y = get_z_test_x_ticks(significat_orthognoal_vs_linear_train, all_x_train, base_ax_values)


        bar_plot_ax_train_props = seaborn_utils.BarPlotAxValues("Test", (0, 1.2), current_ax_bars_train, xticks_labels, set_xticks=base_ax_values)
        bar_plot_ax_test_props = seaborn_utils.BarPlotAxValues("Validation", (0, 1.2), current_ax_bars_test, xticks_labels, set_xticks=base_ax_values,)

        print ("significat_orthognoal_vs_linear_test", significat_orthognoal_vs_linear_test)
        folder_path = os.path.join(output_folder_path,
                                   current_run_name,
                                   str(min_number_of_contacts),
                                   str(current_contact_kind))

        seaborn_utils.output_bar(seaborn_utils.BarPlotProperties(
            bar_plots=[[bar_plot_ax_test_props], [bar_plot_ax_train_props]],
            folder_path=folder_path,
            file_name="cross_dataset_correlation.svg",
            title="bar_plots",
        ),
        marker_values=[
            [scatter_plot_test_x, scatter_plot_test_y]
            # [scatter_plot_train_x, scatter_plot_train_y]
        ]
        )
        a = 3


def test_significant(x1, x2):
    return wilcoxon(x1, x2)

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

    return result_x, result_y


def get_pair_train_test_correlations(feature_subject_1,
                                     feature_subject_2,
                                     current_transformation_matrix,
                                     current_masks,
                                     should_return_permutation):

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
    return current_pair_train_correlations,\
           current_pair_test_correlations,\
           current_pair_permutation_correlation_train,\
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
            mean_value, sem_value = get_pair_image_kind_value(results, number_of_indexes, new_images_kinds, information_name)
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

def get_bar_plot_pair_categories_dataset(results_ratio,
                                         title_relation_graph,
                                         x_labels,
                                         results_sem,
                                         min_max_values
                                         ):

    colors = {
    "Al": "black", # All
    "An": "red", # Animals
    "Bo": "blue", # Body
    "Fa": "cyan", # Face
    "Ho": "brown", # House
    "Pa": "orange", # Patterns
    "Pe": "purple", # People
    "Pl": "pink", # Places
    "To": "gold", # Tool + Tools


    }
    plot_values = []
    x_labels = [[current_label[:2] for current_label in x_label.replace("People", "Faces").split("_")] for x_label in x_labels]

    for current_label_index in range(len(x_labels)):
        current_labels = x_labels[current_label_index]
        each_label_part = results_ratio[current_label_index]
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
            current_label_ax_value = current_label_index
            if current_label_index >= 5:
                current_label_ax_value += 1.5
            plot_values.append(seaborn_utils.BarPlotCategoryValues([current_label_part],
                                                                   color,
                                                                   label="",
                                                                   variance=variance,
                                                                   width=0.5,
                                                                   base_ax_values=[current_label_ax_value],
                                                                   bottom=[bottom]))
            break


    x_labels = ["+".join(current_labels) for current_labels in x_labels]

    x_ticks = np.array(list(range(len(x_labels)))) + np.append(np.zeros(5), (np.ones(len(x_labels) - 5)) + 0.5)
    return seaborn_utils.BarPlotAxValues(title_relation_graph,
                                         min_max_values,
                                         y_label="Relational-linear increase",
                                         x_tick_labels=x_labels,
                                         plot_values=plot_values,
                                         dashed_horizontal_line=1,
                                         set_xticks=x_ticks)


def get_pair_correlations(current_run_name,
                          subject_1_index,
                          subject_2_index,
                          database_name,
                          current_contact_kind,
                          new_images_kinds,
                          current_graph_config,
                          min_number_of_contacts,
                          should_return_permutation=False):
    current_Data = read_data(current_run_name,
                             subject_1_index,
                             subject_2_index,
                             database_name,
                             current_contact_kind,
                             new_images_kinds,
                             current_graph_config.transformation_name,
                             min_number_of_contacts,
                             size_of_test_group=2)

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


def create_lins(current_images_kinds,
                            current_run_name,
                            min_number_of_contacts,
                            database_name):

    print("create_images_histogram ", current_run_name, " ", database_name)
    for current_contact_kind in contacts_to_test:
        all_results = {}

        should_calc_permutation = True

        for graph_config_index, graph_config in enumerate(graph_configs):
            metadata = read_metadata(current_run_name,
                                     database_name,
                                     current_contact_kind,
                                     current_images_kinds,
                                     graph_config.transformation_name,
                                     min_number_of_contacts)
            number_of_indexes = metadata["number_of_indexes"]
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
                

                    current_masks_correlations_train,\
                    current_masks_correlations_test,\
                    current_masks_correlations_permutations_train,\
                    current_masks_correlations_permutations_test = get_pair_correlations(current_run_name,
                                                                                         subject_1_index,
                                                                                         subject_2_index,
                                                                                         database_name,
                                                                                         current_contact_kind,
                                                                                         current_images_kinds,
                                                                                         graph_config,
                                                                                         min_number_of_contacts,
                                                                                         should_calc_permutation)

                    if should_calc_permutation:
                        current_subject_results_train_permutation.append(np.mean(current_masks_correlations_permutations_train))
                        current_subject_results_test_permutation.append(np.mean(current_masks_correlations_permutations_test))

                    current_subject_results_test.append(np.mean(np.array(current_masks_correlations_test)))
                    current_subject_results_train.append(np.mean(np.array(current_masks_correlations_train)))

                if should_calc_permutation:
                    current_contact_kind_permutations_test.append(current_subject_results_test_permutation)
                    current_contact_kind_permutations_train.append(current_subject_results_train_permutation)
                current_graph_results_test.append(current_subject_results_test)
                current_graph_results_train.append(current_subject_results_train)


            
            if should_calc_permutation:
                results = [[
                    seaborn_utils.HeatmapInstance(np.array(current_contact_kind_permutations_test), title="test", min_max_values=(0, 1)),
                    seaborn_utils.HeatmapInstance(np.array(current_contact_kind_permutations_train), title="train", min_max_values=(0, 1))]]
             
             
                should_calc_permutation = False
                all_results["permutation_test"] = current_contact_kind_permutations_test
                all_results["permutation_train"] = current_contact_kind_permutations_train

            
            all_results[graph_config.transformation_name + "_test"] = current_graph_results_test
            all_results[graph_config.transformation_name + "_train"] = current_graph_results_train

        line_plot_results_test = []
        line_plot_results_train = []

        non_diagonal_mask = ~np.eye(len(current_graph_results_test), dtype=bool)
        for (transformation_neuroscience_expression, current_transformation_name) in \
                [("Patterns coding", "permutation")] + [(graph_config.neuroscienec_expression, graph_config.transformation_name) for graph_config in graph_configs]:
            line_plot_results_test.append([transformation_neuroscience_expression, np.array(all_results[current_transformation_name + "_test"])[non_diagonal_mask]])
            line_plot_results_train.append([transformation_neuroscience_expression, np.array(all_results[current_transformation_name + "_train"])[non_diagonal_mask]])


        file_path = os.path.join(output_folder_path,
                                       current_run_name,
                                        curent_database_name,
                                       str(min_number_of_contacts),
                                       str(current_contact_kind),
                                       "_".join([image_kind.name for image_kind in current_images_kinds]),
                                       "line_plots",
                                       "lines_" + "_".join([image_kind.name for image_kind in current_images_kinds]) + ".svg")

        plot_correlates_transformation_lines(line_plot_results_test, line_plot_results_train, file_path, current_images_kinds)


def create_subject_vs_same_subject_sperman_vs_distances(images_kinds,
                                                        current_run_name,
                                                        min_number_of_contacts,
                                                        database_name):
    for current_contact_kind in contacts_to_test:
        metadata = read_metadata(current_run_name,
                                         database_name,
                                         current_contact_kind,
                                         images_kinds[0],
                                         graph_configs[0].transformation_name,
                                         min_number_of_contacts)

        number_of_indexes = metadata["number_of_indexes"]

        for subject_1_index in range(number_of_indexes):
            for subject_2_index in range(number_of_indexes):
                if subject_1_index >= subject_2_index:
                    continue

                # #TODO: rempve (for fast tests)
                # if subject_1_index >= 4 or subject_2_index >= 4:
                #     continue
                results = []
                for current_images_kinds in images_kinds:
                    current_Data = read_data(current_run_name,
                                             subject_1_index,
                                             subject_2_index,
                                             database_name,
                                             current_contact_kind,
                                             current_images_kinds,
                                             graph_configs[0].transformation_name,
                                             min_number_of_contacts,
                                             size_of_test_group=2)
                    feature_subject_1 = current_Data["features_subject_1"]
                    feature_subject_2 = current_Data["features_subject_2"]

                    spearman_matrix_1 = numpy_utils.spearman_matrix(feature_subject_1, feature_subject_1)
                    distances_matrix_1 = numpy_utils.distances_matrix(feature_subject_1, feature_subject_1)

                    spearman_matrix_2 = numpy_utils.spearman_matrix(feature_subject_2, feature_subject_2)
                    distances_matrix_2 = numpy_utils.distances_matrix(feature_subject_2, feature_subject_2)

                    current_images_kinds_str = get_contact_kinds_string(current_images_kinds)
                    results.append([seaborn_utils.HeatmapInstance(map=spearman_matrix_1, title="sujbect_1_spearman_" + current_images_kinds_str, min_max_values=(0, 1)),
                                    seaborn_utils.HeatmapInstance(map=distances_matrix_1, title="subject_1_distances_" + current_images_kinds_str, min_max_values=(0, np.max(distances_matrix_1))),
                                    seaborn_utils.HeatmapInstance(map=spearman_matrix_2, title="sujbect_2_spearman_" + current_images_kinds_str, min_max_values=(0, 1)),
                                    seaborn_utils.HeatmapInstance(map=distances_matrix_2, title="subject_2_distances_" + current_images_kinds_str, min_max_values=(0, np.max(distances_matrix_2)))
                                    ])

                folder_path = os.path.join(output_folder_path,
                                           current_run_name,
                                           str(min_number_of_contacts),
                                           str(current_contact_kind),
                                           "subjcet_vs_same_subject_distances_spearman",
                                           str((subject_1_index, subject_2_index)))

                seaborn_utils.output_heatmap(seaborn_utils.HeatmapProperties(results=results,
                                                                             axes_shape=(len(results), 4),
                                                                             folder_path=folder_path,
                                                                             file_name="Spearman "))




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


def create_same_hemisphere_anatomical_vs_similarity(current_run_name: str,
                                                    min_number_of_contacts: int):
    print("create_same_hemisphere_anatomical_vs_similarity ", current_run_name, ", ", min_number_of_contacts)
    def get_contacts_combos(contact_kind_1, contact_kind_2):
        if ContactKind.Low == contact_kind_1 and ContactKind.Low == contact_kind_2:
            return "both early visual"
        if ContactKind.High == contact_kind_1 and ContactKind.High == contact_kind_2:
            return "both high visual"
        if ContactKind.High == contact_kind_1 and ContactKind.Low == contact_kind_2 or \
                ContactKind.Low == contact_kind_1 and ContactKind.High == contact_kind_2:
            return "one early one high"

        if ContactKind.FaceSelective == contact_kind_1 and ContactKind.FaceSelective == contact_kind_2:
            return "both Face Selective"
        if ContactKind.Low == contact_kind_1 and ContactKind.Low == contact_kind_2:
            return "both early visual"
        if ContactKind.Low == contact_kind_1 and ContactKind.FaceSelective == contact_kind_2 or \
                ContactKind.FaceSelective == contact_kind_1 and ContactKind.Low == contact_kind_2:
            return "one early one high"


    for original_image_kind in ImageKind:
        if original_image_kind == ImageKind.Tool or original_image_kind == ImageKind.Face:
            continue

        graph_config = graph_configs[0]

        results_legend_elements = [[], []]

        for current_contact_kind_index, current_contact_kind in enumerate([ContactKind.All, ContactKind.High]):
            results = [[[]]]

            # for current_contact_kind_index, current_contact_kind in enumerate([ContactKind.All]):
            legend_elements = {}

            number_of_pairs = 0
            number_of_subjects = 0
            number_of_contacts = 0

            for current_ham_index, (current_ham, full_current_ham) in enumerate([("lh", "left hemisphere"), ("rh", "right hemisphere")]):




                for database_name in databases_name:
                    correlations_between_brains = []
                    distances_between_brains = []
                    contacts_combination_category_different_brains = []

                    current_image_kind = original_image_kind



                    if ImageKind.Tools == original_image_kind and database_name != databases_name[0]:
                        current_image_kind = ImageKind.Tool

                    if ImageKind.People == original_image_kind and database_name != databases_name[0]:
                        current_image_kind = ImageKind.Face
                    if [current_image_kind] not in all_images_kinds[database_name]:
                        continue

                    current_images_kinds = [current_image_kind]
                    # current_images_kinds = current_image_kind

                    metadata = read_metadata(current_run_name,
                                             database_name,
                                             current_contact_kind,
                                             current_images_kinds,
                                             graph_config.transformation_name,
                                             min_number_of_contacts)
                    
                    number_of_indexes = metadata["number_of_indexes"]

                    for subject_1_index in range(number_of_indexes):
                        for subject_2_index in range(subject_1_index, number_of_indexes, 1):
                            data = read_data(current_run_name,
                                             subject_1_index,
                                             subject_2_index,
                                             database_name,
                                             current_contact_kind,
                                             current_images_kinds,
                                             graph_config.transformation_name,
                                             min_number_of_contacts,
                                             size_of_test_group=2)
                            features_subject_1 = data["features_subject_1"]
                            features_subject_2 = data["features_subject_2"]


                            subject1_xyz_coordinates = data["subject1_xyz_coordinates"]
                            subject2_xyz_coordinates = data["subject2_xyz_coordinates"]

                            subject1_contacts_types = data["subject1_contacts_types"]
                            subject2_contacts_types = data["subject2_contacts_types"]

                            subject1_contacts_hamesphare = data["subject1_contacts_hamesphare"]
                            subject2_contacts_hamesphare = data["subject2_contacts_hamesphare"]

                            for i in range(features_subject_1.shape[1]):
                                for j in range(features_subject_2.shape[1]):
                                  
                                    try:
                                        if subject1_contacts_hamesphare[i] == current_ham and subject2_contacts_hamesphare[j] == current_ham:
                                            if subject_1_index == subject_2_index:
                                                if i != j:
                                                    pass
                                            else:
                                                if subject1_contacts_types[i] == subject2_contacts_types[j]:
                                                    distances_between_brains.append(np.linalg.norm(subject1_xyz_coordinates[i, :] - subject2_xyz_coordinates[j, :]))
                                                    correlations_between_brains.append(numpy_utils.spearman_matrix(features_subject_1[:, i], features_subject_2[:, j]))
                                                    contacts_combo = get_contacts_combos(subject1_contacts_types[i],
                                                                                         subject2_contacts_types[j])
                                                    contacts_combination_category_different_brains.append(contacts_combo)
                                    except:
                                        pass


                    ham_to_color={
                        "lh": "red",
                        "rh": "blue",

                    }
                    category_to_color = {
                        "both early visual": "red",
                        "both high visual": "darkblue",
                        "one early one high": "seagreen"
                    }



                    hamesphare_to_marker = {
                        "lh": "o",
                        "rh": "s"

                    }

                    dataset_to_marker = {
                        "eventrelatednatural": "o",
                        "blockdesign": "v",
                        "eventrelatedold": "s"
                    }

                    
                    for current_category in sorted(list(set(contacts_combination_category_different_brains))):
                        indexes = [current_pair_category_index for
                                   current_pair_category_index, current_pair_category in
                                   enumerate(contacts_combination_category_different_brains) if current_category == current_pair_category]


                        if current_ham not in legend_elements:
                            pass

                        if current_ham not in legend_elements:

                            legend_elements[current_ham] = Patch(facecolor=ham_to_color[current_ham],
                                                                      label=full_current_ham)


                        X = np.array(filter_indexes(distances_between_brains, indexes))
                        Y = np.array(filter_indexes(correlations_between_brains, indexes))

                        while True:
                            try:
                                coef = np.corrcoef(X.reshape(1, -1), Y.reshape(1, -1))
                                if not coef.shape == (2, 2):
                                    raise
                                current_category_correlation = coef[0, 1]
                                break
                            except:
                                a = 3

                        if current_contact_kind == ContactKind.All:
                            title = "Early and High visual Cortex"
                        if current_contact_kind == ContactKind.High:
                            title = "High visual Cortex"
                        if current_contact_kind == ContactKind.FaceSelective:
                            title = "Face Selective"
                        if current_contact_kind == ContactKind.LowAndFaceSelective:
                            title = "Early and Face Selective visual Cortex"

                        results[0][0].append(seaborn_utils.ScatterPlotInstance(x=X,
                                                                               y=Y,
                                                                                x_label="Pair-wise contact anatomical distance",
                                                                                y_label="Pair-wise contact tuning similarity (Spearman \u03C1)",
                                                                                rgb_values=None,
                                                                                color=ham_to_color[current_ham],
                                                                                label=current_category + ". \u03C1: %.2f" % current_category_correlation,
                                                                                x_min_max_values=(0, 1),
                                                                                y_min_max_values=(-1.25, 1.25)))

            results_legend_elements[0].append(list(legend_elements.values()))
            set_scatter_mi_max_valeus(results)
            pearson_values = get_pearson_of_scatter(results)

            line_props = None
            if len(results[0][0]):
                line_props = get_regression_for_scatter(results)
            else:
                a = 3

            seaborn_utils.output_scatter(seaborn_utils.ScatterPlotProperties(results=results,
                                                                             should_plot_regression=False,
                                                                             axes_shape=(1, len(results[0])),
                                                                             folder_path=os.path.join(output_folder_path,
                                                                                                      current_run_name,
                                                                                                      # database_name,
                                                                                                      str(min_number_of_contacts),
                                                                                                      get_contact_kinds_string(current_images_kinds),
                                                                                                      "scatter_distances",
                                                                                                      str(current_contact_kind)),
                                                                                 file_name="distances_vs_correlartion_scatter",
                                                                             legend_elements=results_legend_elements,
                                                                             plot_avareges_bars=False,
                                                                             texts=pearson_values,
                                                                             fig_title=get_contact_kinds_string(current_images_kinds)),
                                         line_props=line_props)

def get_flatten_valeus(results):
    flatten_results = []
    for current_row in results:
        current_row_results = []
        for current_ax_results in current_row:
            x_values = []
            y_valeus = []
            for current_graph_values in current_ax_results:
                x_values += current_graph_values.x.flatten().tolist()
                y_valeus += current_graph_values.y.flatten().tolist()
            current_row_results.append({"X": x_values,
                                        "Y": y_valeus})
        flatten_results.append(current_row_results)
    return flatten_results

def get_max_min_for_scatter_line(flatten_values):
    try:
        min_value = min(flatten_values[0][0]["X"])
        max_value = max(flatten_values[0][0]["X"])
    except:
        return 0, 0
    for current_row in flatten_values:
        for current_ax_results in current_row:
            if min(current_ax_results["X"]) < min_value:
                min_value = min(current_ax_results["X"])
            if max(current_ax_results["X"]) > max_value:
                max_value = max(current_ax_results["X"])
    return min_value, max_value



def get_regression_for_scatter(results):
    line_props_results = []
    flatten_values = get_flatten_valeus(results)
    min_x_value, max_x_value = get_max_min_for_scatter_line(flatten_values)
    for row_flatten_values in flatten_values:
        curent_row_results = []
        try:
            for current_flatten_values in row_flatten_values:
                coef = np.polyfit(current_flatten_values["X"], current_flatten_values["Y"], 1)
                poly1d_fn = np.poly1d(coef.flatten())
                xseq = np.linspace(min_x_value, max_x_value, num=20)
                curent_row_results.append([seaborn_utils.GraphIterationValues(xseq,
                                                                              poly1d_fn(xseq),
                                                                              color="black",
                                                                              width=10,
                                                                              text=f"regression slope: {round(coef[0], 4)}")])
        except:
            a = 3


        line_props_results.append(curent_row_results)
    return line_props_results



def get_line_props_for_scatter(results):
    lines_results = []
    for current_contact_result in results:
        current_contact_lines = []
        for current_current_ham_result in current_contact_result:
            current_ax_lines = []

            x_values = np.concatenate([value.x for value in current_current_ham_result])
            y_values = np.concatenate([value.y for value in current_current_ham_result])
            bin_size = 0.2
            min_bin_value = 0
            for i in range(3):

                max_bin_value = min_bin_value + bin_size
                indexes_of_bin = np.where(np.logical_and(x_values >= min_bin_value, x_values <= max_bin_value))
                values_of_bin = y_values[indexes_of_bin]

                if values_of_bin.shape[0] == 0:
                    continue

                bin_avg = np.mean(values_of_bin)

                bin_std = np.std(values_of_bin)
                current_ax_lines.append(seaborn_utils.GraphIterationValues(x_values= [min_bin_value, max_bin_value],
                                                                           y_values=[bin_avg, bin_avg],
                                                                           variances=[(min_bin_value + max_bin_value) / 2,
                                                                                      bin_avg,
                                                                                      bin_std],
                                                                           color="black",
                                                                           text=None,
                                                                           width=None))
                min_bin_value = max_bin_value
            current_contact_lines.append(current_ax_lines)
        lines_results.append(current_contact_lines)
    return lines_results

def get_pearson_of_scatter(results):
    pearson_results = []
    for current_contact_result in results:
        current_contact_pearson = []
        for current_current_ham_result in current_contact_result:
            if len(current_current_ham_result) == 0:
                current_contact_pearson.append("no values")
            else:
                x_values = np.concatenate([value.x for value in current_current_ham_result])
                y_values = np.concatenate([value.y for value in current_current_ham_result])
                while True:
                    try:
                        statistic, p_value = pearsonr(x_values.flatten(), y_values.flatten())
                        break
                    except:
                        a = 3

                current_contact_pearson.append("r=" + str(round(statistic, 5)) + ". p_value=" + str(round(p_value, 3)))

        pearson_results.append(current_contact_pearson)
    return pearson_results

def get_line_plot_scatter_regression(distances, correlations):
    correlation_value = np.corrcoef(distances, correlations)[0,1]
    coef = np.polyfit(distances, correlations, 1)
    poly1d_fn = np.poly1d(coef)
    xseq = np.linspace(np.min(distances), np.max(distances), num=20)
    return seaborn_utils.GraphIterationValues(
        x_values = xseq,
        y_values = poly1d_fn(xseq),
        color="black",
        width=5,
        line_title="All contacts. \u03C1: %.2f" % correlation_value
    )

def set_scatter_mi_max_valeus(results):
    min_x = 10000
    min_y = 10000
    max_x = -10000
    max_y = -10000
    for results_arr in results:
        for current_results in results_arr:
            for current_result in current_results:
                if current_result.x.shape[0] == 0:
                    continue
                if min_x > np.min(current_result.x):
                    min_x = np.min(current_result.x)
                if max_x < np.max(current_result.x):
                    max_x = np.max(current_result.x)
                if min_y > np.min(current_result.y):
                    min_y = np.min(current_result.y)
                if max_y < np.max(current_result.y):
                    max_y = np.max(current_result.y)

    for results_arr in results:
        for current_results in results_arr:
            for current_result in current_results:
                current_result.x_min_max_values = (min_x, max_x)
                current_result.xy_min_max_values = (min_y, max_y)



def filter_indexes(values, indexes):
    return [current_value for current_value_index, current_value in enumerate(values) if current_value_index in indexes]


graph_configs: List[GraphConfig] = [
    # GraphConfig(transformation_name="orthogonal_normalize"),
        GraphConfig(transformation_name="orthogonal", neuroscienec_expression="Relational coding"),
        GraphConfig(transformation_name="linear", neuroscienec_expression="Linear coding")
    # GraphConfig(transformation_name="linear_normalized")
]


def get_correlation_value(first_subject_features, second_subject_features, transformation_matrix=None, mask=None, use_distances=False):
    if mask is not None:
        first_subject_features = first_subject_features[mask].reshape(-1, first_subject_features.shape[1])
        second_subject_features = second_subject_features[mask].reshape(*first_subject_features.shape)
    if use_distances:
        return get_distance_corelation_value(first_subject_features, second_subject_features, transformation_matrix=transformation_matrix)

    return get_spearman_corelation_value(first_subject_features, second_subject_features, transformation_matrix=transformation_matrix)


def plot_correlates_transformation_lines(line_plot_results_test, line_plot_results_train, file_path, current_images_kinds):
    fig_title = "_".join([current_images_kind_.name for current_images_kind_ in current_images_kinds])
    current_graph_results_ordered_test = []
    current_graph_results_ordered_train = []
    colors = ["red", "green", "blue"]


    for i, current_line_plot_result_test in enumerate(line_plot_results_test):
        current_line_plot_result_train = line_plot_results_train[i]
        # current_graph_results_permutation.append(seaborn_utils.GraphIterationValues(np.arange(order_permutation.shape[0]),
        #                                                                             current_line_plot_result[1][order_permutation],
        #                                                                             line_title=current_line_plot_result[0],
        #                                                                             variances=None))
        current_graph_results_ordered_test.append(seaborn_utils.GraphIterationValues(np.arange(current_line_plot_result_test[1].shape[0]),
                                                                                np.sort(current_line_plot_result_test[1]),
                                                                                line_title=current_line_plot_result_test[0],
                                                                                variances=None,
                                                                                color=colors[i]))

        current_graph_results_ordered_train.append(
            seaborn_utils.GraphIterationValues(np.arange(current_line_plot_result_train[1].shape[0]),
                                               np.sort(current_line_plot_result_train[1]),
                                               line_title=current_line_plot_result_train[0],
                                               variances=None,
                                               color=colors[i]))

    seaborn_utils.save_graph([
        seaborn_utils.OutputGraphPropertied(current_graph_results_ordered_test,
                                            x_label="Pairs of subjects",
                                            y_label="Inter-subject correlation (Spearman)",
                                            title="Validation",
                                            file_path=file_path,
                                            max_y_lim=1.4,
                                            min_y_lim=-0.0,
                                            show_legend=False,
                                            x_labels=None,
                                            text="B"),
        seaborn_utils.OutputGraphPropertied(current_graph_results_ordered_train,
                                            x_label="Pairs of subjects",
                                            y_label="Inter-subject correlation (Spearman)",
                                            title="Train",
                                            file_path=file_path,
                                            max_y_lim=1.4,
                                            min_y_lim=-0.0,
                                            show_legend=True,
                                            x_labels=None,
                                            text="A")
          ],
        fig_title)



def get_distance_corelation_value(first_subject_features, second_subject_features, transformation_matrix=None):
    if transformation_matrix is not None:
        return np.mean(np.diag(cdist(first_subject_features @ transformation_matrix, second_subject_features)))
    return np.mean(np.diag(cdist(first_subject_features, second_subject_features)))

def get_spearman_corelation_value(first_subject_features, second_subject_features, transformation_matrix=None):
    if transformation_matrix is not None:
        return np.mean(np.diag(numpy_utils.spearman_matrix(first_subject_features @ transformation_matrix, second_subject_features)))
    return np.mean(np.diag(numpy_utils.spearman_matrix(first_subject_features, second_subject_features)))

def get_pearson_corelation_value(first_subject_features, second_subject_features, transformation_matrix=None):
    if transformation_matrix is not None:
        return np.mean(np.diag(numpy_utils.pearson_matrix(first_subject_features @ transformation_matrix, second_subject_features)))
    return np.mean(np.diag(numpy_utils.pearson_matrix(first_subject_features, second_subject_features)))


min_number_of_contacts = 4
output_folder_path = "./output/test_results"
databases_name = ["eventrelatednatural", "blockdesign", "eventrelatedold"]
contacts_to_test = [ContactKind.High]

for starting_run_name in ["spearman_alignment"]:

    create_cross_dataset_bar_plots(starting_run_name, min_number_of_contacts)
    
    create_same_hemisphere_anatomical_vs_similarity(starting_run_name,
                                                    min_number_of_contacts)
    
    for curent_database_name in databases_name:
        images_kinds = all_images_kinds[curent_database_name]
        brain_data = BrainData(curent_database_name)
        print("current_database " + curent_database_name)

        starting_time = datetime.now()
                
        create_graph_files_all_kinds(starting_run_name, graph_configs, contacts_to_test, curent_database_name)
        print("finish running " + str(datetime.now() - starting_time))





