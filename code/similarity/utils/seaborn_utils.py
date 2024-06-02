import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from scipy.spatial.distance import cdist
from scipy import stats
from typing import Optional
import warnings
from scipy.interpolate import make_interp_spline, BSpline

warnings.filterwarnings("ignore", message="Polyfit may be poorly conditioned")


@dataclass
class IterationBoxPlotProperties:
    X: np.array
    title: str
    min_max_values: Optional[Tuple[float, float]]


class BarPlotCategoryValues:
    def __init__(self,
                 X: np.array,
                 color: str,
                 label: str,
                 variance: Optional[float],
                 width: Optional[float] = None,
                 base_ax_values: List[float] = None,
                 bottom: float = 0):
        self.X = X
        self.base_ax_values = base_ax_values
        self.color = color
        self.label = label
        self.variance = variance
        self.width = width
        self.bottom = bottom


class BarPlotAxValues:
    def __init__(self,
                 title: str,
                 min_max: np.array,
                 plot_values: List[BarPlotCategoryValues],
                 x_tick_labels: List[str],
                 dashed_horizontal_line: float = None,
                 y_label: str = None,
                 set_xticks: List[float] = None,
                 should_show_legend=True):
        self.title = title
        self.min_max = min_max
        self.plot_values = plot_values
        self.x_tick_labels = x_tick_labels
        self.set_xticks = set_xticks
        self.y_label = y_label
        self.dashed_horizontal_line = dashed_horizontal_line
        self.should_show_legend = should_show_legend


@dataclass
class BarPlotProperties:
    bar_plots: List[List[BarPlotAxValues]]
    folder_path: str
    file_name: str
    title: str


@dataclass
class BoxPlotProperties:
    box_plots: List[List[IterationBoxPlotProperties]]
    axes_shape: Tuple[int, int]
    folder_path: str
    file_name: str


class OutputGraphPropertied:
    def __init__(self, graph_iteration_values, x_label, y_label, title, file_path, max_y_lim=1, min_y_lim=0,
                 show_legend=True, x_labels=None, text=None):
        self.graph_iteration_values = graph_iteration_values
        self.x_label = x_label
        self.y_label = y_label
        self.title = title
        self.file_path = file_path
        self.max_y_lim = max_y_lim
        self.min_y_lim = min_y_lim
        self.show_legend = show_legend
        self.x_labels = x_labels
        self.text = text


class GraphIterationValues:
    def __init__(self, x_values, y_values, line_title=None, variances=None, color=None, text=None, width=None):
        self.x_values = x_values
        self.y_values = y_values
        self.line_title = line_title
        self.variances = variances
        self.color = color
        self.text = text
        self.width = width


class ScatterPlotInstance:
    def __init__(self, x,
                 y,
                 x_label,
                 y_label,
                 rgb_values,
                 title=None,
                 min_max_values=None,
                 label=None,
                 color=None,
                 x_min_max_values=None,
                 y_min_max_values=None,
                 marker: str = "o"):
        self.x = x
        self.y = y
        self.x_label = x_label
        self.y_label = y_label
        self.rgb_values = rgb_values
        self.title = title
        self.min_max_values = min_max_values
        self.label = label
        self.color = color
        self.x_min_max_values = x_min_max_values
        self.y_min_max_values = y_min_max_values
        self.marker = marker


class ScatterPlotProperties:
    def __init__(self,
                 results,
                 axes_shape,
                 folder_path,
                 file_name,
                 color_pallete=sns.color_palette("Spectral", as_cmap=True),
                 should_plot_regression=False,
                 should_plot_diagonal_line=False,
                 plot_avareges_bars=False,
                 legend_elements: any = None,
                 texts: List[List[str]] = None,
                 fig_title: str = None):
        self.results = results
        self.axes_shape = axes_shape
        self.folder_path = folder_path
        self.file_name = file_name
        self.color_pallete = color_pallete
        self.should_plot_regression = should_plot_regression
        self.should_plot_diagonal_line = should_plot_diagonal_line
        self.plot_avareges_bars = plot_avareges_bars
        self.legend_elements = legend_elements
        self.texts = texts
        self.fig_title = fig_title


class HeatmapInstance:
    def __init__(self, map, title, min_max_values=None, x_tick_label="auto", y_tick_label="auto"):
        self.map = map
        self.title = title
        self.min_max_values = min_max_values
        self.x_tick_label = x_tick_label
        self.y_tick_label = y_tick_label


class HeatmapProperties:
    def __init__(self, results, axes_shape, folder_path, file_name):
        self.results = results
        self.axes_shape = axes_shape
        self.folder_path = folder_path
        self.file_name = file_name


def save_heatmap(matrix, file_path, x_label, y_label, title):
    plt.figure()
    sns_plot = sns.heatmap(matrix)
    figure = sns_plot.get_figure()
    plt.title(title, fontsize=40)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    figure.savefig(file_path)
    plt.close()


def save_graph(properties: List[OutputGraphPropertied], fig_title=None):
    fig, axes = plt.subplots(
        len(properties),
        1,
        figsize=(1 * 25, len(properties) * 12.5),
        constrained_layout=True)


    for graph_index, graph_properties in enumerate(properties):
        try:
            ax = axes[graph_index]
        except:
            ax = axes
        for iteration_index, iteration_values in enumerate(graph_properties.graph_iteration_values):

            ax.plot(iteration_values.x_values,
                    iteration_values.y_values,
                    label=iteration_values.line_title,
                    color=iteration_values.color,
                    linewidth=10)
            if iteration_values.variances is not None:
                ax.errorbar(iteration_values.x_values, iteration_values.y_values, yerr=iteration_values.variances,
                            fmt='o', ecolor='red', color='yellow')

        if graph_properties.x_labels is not None:
            ax.set_xticks(range(len(graph_properties.x_labels)))
            ax.set_xticklabels(graph_properties.x_labels)

        if graph_properties.text is not None:
            ax.text(0.01, 1.2, graph_properties.text, horizontalalignment="left", fontsize=40)

        if graph_properties.show_legend:
            ax.legend(fontsize=35, prop={'size': 30}, loc="lower right")

        ax.set_xlabel(graph_properties.x_label, fontsize=45)
        ax.set_ylabel(graph_properties.y_label, fontsize=45)
        if graph_properties.title is not None:
            ax.text((len(iteration_values.x_values) - 1) / 2, 1.1, graph_properties.title, horizontalalignment="center", fontsize=80, weight='bold')
            # ax.set_title(graph_properties.title, fontsize=50)
            ax.set_title("  ", fontsize=70)
        ax.set_ylim(graph_properties.min_y_lim, graph_properties.max_y_lim)
        ax.tick_params(axis='both', which='major', labelsize=40, length=20, width=3, color="black")
    # if fig_title != None:
    #     # fig.suptitle(fig_title, fontsize=70)

    file_dir_path = os.path.dirname(graph_properties.file_path)
    if not os.path.exists(file_dir_path):
        os.makedirs(file_dir_path)
    plt.savefig(graph_properties.file_path, format="svg")

    plt.close()


#


def output_current_scatter_ax(current_result, ax, should_plot_regression, should_plot_diagonal_line,
                              plot_avareges_bars):
    amount_of_images = current_result.x.shape[0]
    opacity = 0.2

    if amount_of_images <= 1000:
        opacity = 0.5
    if amount_of_images <= 300:
        opacity = 1

    if current_result.x.shape[0] == 0:
        return

    # ax.set_title(current_result.title, fontsize=60)
    while True:
        try:
            mat = np.stack([current_result.x.flatten(), current_result.y.flatten()]).T
            break
        except:
            raise
            a = 3
    if current_result.rgb_values is None:
        colors = current_result.color
    else:
        colors = current_result.rgb_values

    if current_result.min_max_values is None:
        vmin, vmax = None, None
    else:
        vmin, vmax = current_result.min_max_values
    x = mat[:, 0]
    y = mat[:, 1]
    size = [200] * y.shape[0]

    ax.scatter(x=x,
               y=y,
               color=colors,
               s=size,
               alpha=opacity,
               vmin=vmin,
               vmax=vmax,
               label=current_result.label,
               marker=current_result.marker)

    if current_result.label is not None:
        ax.legend()

    if should_plot_regression:
        num_parts = 1
        min_part_value = np.min(x)
        for current_part in range(num_parts):
            max_part_value = (1 + current_part) * np.max(x) / num_parts
            mask = (min_part_value < x) & (x < max_part_value)
            if mask.any():
                coef = np.polyfit(x[mask], y[mask], 1)
                poly1d_fn = np.poly1d(coef)
                xseq = np.linspace(min_part_value, max_part_value, num=20)
                ax.plot(xseq, poly1d_fn(xseq), linewidth=5)
            min_part_value = max_part_value

    if plot_avareges_bars:
        num_parts = 4
        min_part_value = np.min(x)
        for current_part in range(num_parts):
            max_part_value = (1 + current_part) * np.max(x) / num_parts
            mask = (min_part_value < x) & (x < max_part_value)
            if mask.any():
                mean_value = np.mean(y[mask])
                xseq = np.array([min_part_value, min_part_value, max_part_value, max_part_value])
                yseq = np.array([0, mean_value, mean_value, 0])
                ax.plot(xseq, yseq)
            min_part_value = max_part_value

    spearman_correlation_value, spearman_correlation_p_value = stats.spearmanr(x, y)
    text = f"spearman value: {spearman_correlation_value}. p value: {spearman_correlation_p_value}"
    # ax.text(0.85,
    #         0.85,
    #         text,
    #         fontsize=20)  # add text

    if current_result.x_min_max_values is None:
        x_min, x_max = np.min(x), np.max(x)
    else:
        x_min, x_max = current_result.x_min_max_values

    if current_result.y_min_max_values is None:
        y_min, y_max = np.min(y), np.max(y)
    else:
        y_min, y_max = current_result.y_min_max_values

    ax.set_ylim(y_min, y_max)
    ax.set_xlim(x_min, x_max)

    if should_plot_diagonal_line:
        ax.plot([x_min, x_max], [y_min, y_max], 'b--')

    ax.set_xlabel(current_result.x_label, fontsize=30)
    ax.set_ylabel(current_result.y_label, fontsize=30)


def output_scatter(properties: ScatterPlotProperties, line_props: List[List[List[GraphIterationValues]]] = None):
    plt.rcParams["axes.linewidth"] = 2.50

    fig, axes = plt.subplots(
        properties.axes_shape[0],
        properties.axes_shape[1],
        figsize=(properties.axes_shape[1] * 35, properties.axes_shape[0] * 20))

    for i in range(len(properties.results)):
        for j in range(len(properties.results[0])):
            if properties.axes_shape[0] == 1:
                if properties.axes_shape[1] == 1:
                    ax = axes
                else:
                    ax = axes[j]
            else:
                if properties.axes_shape[1] == 1:
                    ax = axes[i]
                else:
                    ax = axes[i][j]
            current_ax_results = properties.results[i][j]
            text = ""

            current_legend_elements = None
            if properties.legend_elements is not None and properties.legend_elements[i] is not None:
                current_legend_elements = properties.legend_elements[i][j]

            if type(current_ax_results) is list:
                for current_results in current_ax_results:
                    output_current_scatter_ax(current_results,
                                              ax,
                                              properties.should_plot_regression,
                                              properties.should_plot_diagonal_line,
                                              properties.plot_avareges_bars)

                if line_props != None:
                    current_line_plots = line_props[i][j]
                    linewidth = 10
                    for current_line_plot in current_line_plots:
                        if current_line_plot.width != None:
                            linewidth = current_line_plot.width

                        ax.plot(current_line_plot.x_values,
                                current_line_plot.y_values,
                                linewidth=linewidth,
                                color=current_line_plot.color)
                        if current_line_plot.text != None:
                            text += current_line_plot.text + "\n"

                        if current_line_plot.variances is not None:
                            ax.errorbar(current_line_plot.variances[0],
                                        current_line_plot.variances[1],
                                        yerr=current_line_plot.variances[2],
                                        fmt='o',
                                        elinewidth=10,
                                        ecolor='red',
                                        color='yellow')

                    # if current_line_plots.text != None:
                    #     ax.text(1, 0.1, current_line_plots.text, fontsize=30)

                ax.set_xlabel(ax.get_xlabel(), fontname="Arial", fontsize=40)
                ax.set_ylabel(ax.get_ylabel(), fontname="Arial", fontsize=40)
                ax.tick_params(axis='both', which='major', labelsize=30, length=20, width=3, color="black")


            else:
                output_current_scatter_ax(current_ax_results,
                                          ax,
                                          properties.should_plot_regression,
                                          properties.should_plot_diagonal_line,
                                          properties.plot_avareges_bars)
            # ax.legend(fontsize=40,  frameon=False)
            if current_legend_elements != None:
                ax.legend(fontsize=40, handles=current_legend_elements)
            else:
                ax.legend(fontsize=40)

            if properties.texts != None and properties.texts[i] != None:
                ax.text(5, 1, text + "\n" + properties.texts[i][j], horizontalalignment="left", fontsize=30)
    # if properties.fig_title != None:
    #     fig.suptitle(properties.fig_title, fontsize=70)

    path = os.path.join(properties.folder_path, properties.file_name + ".svg")
    if not os.path.exists(properties.folder_path):
        os.makedirs(properties.folder_path)
    fig.savefig(path, format="svg")
    plt.close(fig)


def output_heatmap(properties: HeatmapProperties, use_log_scale=False):
    # gs_kw = dict(width_ratios=widths, height_ratios=heights)

    fig, axes = plt.subplots(
        properties.axes_shape[0],
        properties.axes_shape[1],
        figsize=(len(properties.results[0] * 25),
                 len(properties.results) * 12.5))

    # fig, axes = plt.subplots(
    #     properties.axes_shape[0],
    #     properties.axes_shape[1],
    #     figsize=(50, 50))

    for i in range(properties.axes_shape[0]):
        for j in range(properties.axes_shape[1]):
            current_result = properties.results[i][j]
            if properties.axes_shape[0] == 1:
                ax = axes[j]
            else:
                ax = axes[i][j]

            current_map = current_result.map
            mean = round(np.mean(current_map), 2)
            std = round(np.std(current_map), 2)
            text = 'mean value: ' + str(mean) + ", std value: " + str(std)

            if current_map.shape[0] == current_map.shape[1]:
                flatten_map_without_diag = current_map[~np.eye(current_map.shape[0], dtype=bool)]
                mean_without_diag = round(np.mean(flatten_map_without_diag), 2)
                std_without_diag = round(np.std(flatten_map_without_diag), 2)
                text += '.\nmean_without_diag value: ' + str(mean_without_diag) + ",std_without_diag value: " + str(
                    std_without_diag)

                # flatten_map_without_diag_and_minus_1 = flatten_map_without_diag[flatten_map_without_diag > -1]
                # mean_without_diag_and_minus_1 = round(np.mean(flatten_map_without_diag_and_minus_1), 2)
                # std_without_diag_and_minus_1 = round(np.std(flatten_map_without_diag_and_minus_1), 2)

                # '.\n mean_without_diag_and_minus_1: ' +  str(mean_without_diag_and_minus_1) + ",std_without_diag_and_minus_1 value: " + str(std_without_diag_and_minus_1),

            ax.text(0.85,
                    0.85,
                    text,
                    fontsize=20)  # add text
            ax.set_title(current_result.title, fontsize=40)

            # if current_result.x_tick_label is not None:
            #     ax.set_xticks(range(len(current_result.x_tick_label)))
            #     ax.set_xticklabels(current_result.x_tick_label)
            #
            # if current_result.y_tick_label is not None:
            #     ax.set_yticks(range(len(current_result.y_tick_label)))
            #     ax.set_yticklabels(current_result.y_tick_label)

            if use_log_scale:
                current_map = np.log(current_map - np.min(current_map) + 0.0001)
            if current_result.min_max_values is None:
                vmin, vmax = None, None
            else:
                vmin, vmax = current_result.min_max_values
            sns.heatmap(current_map,
                        ax=ax,
                        vmin=vmin,
                        vmax=vmax,
                        cmap=sns.color_palette("vlag", as_cmap=True),
                        xticklabels=current_result.x_tick_label,
                        yticklabels=current_result.y_tick_label)

            # print np file as text
            if not os.path.exists(properties.folder_path):
                os.makedirs(properties.folder_path)
            np_text_file_path = os.path.join(
                properties.folder_path, current_result.title + ".txt")
            np.savetxt(np_text_file_path, current_result.map)

    file_name = properties.file_name
    if not file_name.endswith(".jpeg"):
        file_name += ".jpeg"
    path = os.path.join(properties.folder_path, file_name)
    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def output_box_plot(properties: BoxPlotProperties):
    fig, axes = plt.subplots(
        properties.axes_shape[0],
        properties.axes_shape[1],
        figsize=(len(properties.box_plots[0] * 25),
                 len(properties.box_plots) * 12.5))

    for i in range(properties.axes_shape[0]):
        for j in range(properties.axes_shape[1]):
            current_result = properties.box_plots[i][j]
            if properties.axes_shape[0] == 1:
                ax = axes[j]
            else:
                ax = axes[i][j]

            X_values = current_result.X

            ax.boxplot(X_values)
            ax.set_ylim(current_result.min_max_values[0], current_result.min_max_values[1])

            # print np file as text
            if not os.path.exists(properties.folder_path):
                os.makedirs(properties.folder_path)
            np_text_file_path = os.path.join(
                properties.folder_path, current_result.title + ".txt")
            np.savetxt(np_text_file_path, X_values)

    file_name = properties.file_name
    if not file_name.endswith(".jpeg"):
        file_name += ".jpeg"
    path = os.path.join(properties.folder_path, file_name)
    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def get_scatter_colors_indexes(tsne_mat):
    min_x = min([val[0] for val in tsne_mat])
    min_y = min([val[1] for val in tsne_mat])
    distances = [(i, np.linalg.norm(val - [min_x, min_y]))
                 for i, val in enumerate(tsne_mat)]
    sorted_distances = sorted(distances, key=lambda x: x[1])
    result = [0] * len(distances)
    for i, (sorted_index, _) in enumerate(sorted_distances):
        result[sorted_index] = i

    return result


def get_scatter_colors_rgb(data):
    min_values = np.min(data, axis=0)
    max_values = np.max(data, axis=0)

    normalized_values = np.nan_to_num(np.divide(data - min_values,
                                                max_values - min_values))

    red_values = (normalized_values[:, 0]).reshape(-1, 1)
    green_values = (normalized_values[:, 1]).reshape(-1, 1)
    blue_values = 2 - red_values - green_values
    blue_values /= np.max(blue_values)

    return np.concatenate((red_values, green_values, blue_values), axis=1)


def output_bar(properties: BarPlotProperties, marker_values: any = None):
    num_rows = len(properties.bar_plots)
    num_cols = len(properties.bar_plots[0])
    fig, axes = plt.subplots(num_rows,
                             num_cols,
                             figsize=(num_cols * 80,
                                      num_rows * 30))

    for i, _ in enumerate(properties.bar_plots):
        for j, _ in enumerate(properties.bar_plots[0]):
            if len(properties.bar_plots) == 1:
                if len(properties.bar_plots[0]) == 1:
                    ax = axes
                else:
                    ax = axes[j]
            elif len(properties.bar_plots[0]) == 1:
                ax = axes[i]
            else:
                ax = axes[i][j]

            width = 0.25  # the width of the bars
            current_bar = properties.bar_plots[i][j]
            for current_bar_plot_properties_index, current_bar_plot_properties in enumerate(current_bar.plot_values):
                # x_indexes = np.arange(len(current_bar_plot_properties.X))
                if current_bar_plot_properties.width is not None:
                    width = current_bar_plot_properties.width

                X_axis_location = current_bar_plot_properties.base_ax_values
                try:
                    rects = ax.bar(X_axis_location,
                                   current_bar_plot_properties.X,
                                   width,
                                   yerr=current_bar_plot_properties.variance,
                                   error_kw=dict(lw=5, capsize=5, capthick=3),
                                   label=current_bar_plot_properties.label,
                                   color=current_bar_plot_properties.color,
                                   bottom=current_bar_plot_properties.bottom)


                except:
                    a = 3
            if current_bar.dashed_horizontal_line is not None:
                ax.axhline(y=current_bar.dashed_horizontal_line, color='r', linestyle='dashed')

            if current_bar is not None:
                ax.set_title(current_bar.title, fontsize=50)

            if current_bar.y_label != None:
                ax.set_ylabel(current_bar.y_label, fontsize=70)
            if marker_values is not None:
                if len(marker_values) > i :
                    for current_marker_values_index in range(len(marker_values[i][0])):
                        current_marker_locations = marker_values[i][0][current_marker_values_index]
                        current_marker_values = marker_values[i][1][current_marker_values_index]

                        ax.scatter(
                            [current_marker_location for current_marker_location in current_marker_locations],
                            current_marker_values,
                            marker="*",
                            s=[600] * len(current_marker_locations),
                            color="red")
            ax.set_ylim(current_bar.min_max[0], current_bar.min_max[1])
            # ax.set_title(current_bar.title, fontsize=60)

            ax.set_xticks(current_bar.set_xticks)
            ax.set_xticklabels(current_bar.x_tick_labels, fontsize=60, rotation=45)
            ax.yaxis.set_tick_params(labelsize=60)
            ax.tick_params(axis="both", direction="in", pad=15)
            # ax.margins(0.5, 5)
            if current_bar.should_show_legend:
                ax.legend(fontsize=45)


    # fig.suptitle(properties.title, fontsize=70)

    file_path = os.path.join(properties.folder_path, properties.file_name)

    file_dir_path = os.path.dirname(file_path)
    if not os.path.exists(file_dir_path):
        os.makedirs(file_dir_path)
    plt.savefig(file_path, format="svg")

    plt.close(fig)

# iteration_values = [GraphIterationValues([1, 2, 3], [0.1, 0.2, 0.3], "first line title"),
#                     GraphIterationValues(
#                         [1, 2, 3], [0.2, 0.3, 0.4], "second line title"),
#                     GraphIterationValues(
#                         [1, 2, 3], [0.3, 0.4, 0.5], "third line title"),
#                     GraphIterationValues([1, 2, 3], [0.4, 0.5, 0.6], "fourth line title")]

# properties = OutputGraphPropertied(
#     iteration_values, "x label", "y label", "plot title", "./output.jpeg")
# save_graph(properties)
