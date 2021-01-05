"""Plot function definitions."""

import numpy as np
import matplotlib.cm
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn


def plot_battery_cases_comparisons_bubbles(
        read_path,
        results_path,
        plot_type
        # Choices: 'simple_payback_time', 'discounted_payback_time', 'storage_capacity', 'efficiency', 'energy_cost',
        # 'operation_cost_savings_annual', 'operation_cost_savings_annual_percentage'.
):
    """Plot various battery case comparisons."""

    # Define plot settings.
    seaborn.set()
    plt.rcParams['font.serif'] = 'Arial'
    plt.rcParams['font.family'] = 'serif'

    # Load result data for plotting.
    results = pd.read_csv(read_path, index_col='battery_technology')
    set_years = results.columns
    set_battery_technologies = results.index
    x_array = np.arange(1, set_years.shape[0]+1, 1)
    y_array = np.arange(1, set_battery_technologies.shape[0]+1, 1)
    colors = matplotlib.cm.Paired(np.linspace(0, 1, set_battery_technologies.shape[0]))

    # Create plot.
    (fig, ax) = plt.subplots(1, 1)
    for i_battery_technology in np.arange(0, set_battery_technologies.shape[0], 1):
        for i_year in np.arange(0, len(x_array), 1):
            ax.scatter(
                x_array[i_year],
                y_array[i_battery_technology],
                marker='o', facecolors=colors[i_battery_technology], edgecolors='none',
                s=(
                    2000.0/max(results.max(axis=0)) * results.iloc[i_battery_technology, i_year]
                    if max(results.max(axis=0)) != 0.0 else 0.0
                ),
                alpha=0.7
            )

            if results.iloc[i_battery_technology, i_year] != 0.0:
                ax.text(
                    x_array[i_year], y_array[i_battery_technology],
                    (
                        format(
                            results.iloc[i_battery_technology, i_year],
                            ('.0f' if plot_type != 'efficiency' and plot_type != 'savings_year_percentage' else '.2f')
                        )
                        + ('%' if plot_type == 'savings_year_percentage' else '')
                    ),
                    weight='bold',
                    fontsize=9,
                    bbox={'facecolor': 'none', 'alpha': 0.5, 'pad': 1, 'edgecolor': 'none'},
                    ha='center', va='center'
                )

    # Modify plot.
    ax.set_xticks(x_array)
    x_labels = [item.get_text() for item in ax.get_xticklabels()]
    for i_year in np.arange(0, len(x_array), 1):
        x_labels[i_year] = set_years[i_year]
    ax.set_xticklabels(x_labels)
    legend_labels = ['Flooded LA', 'VRLA', 'LFP', 'NCA', 'NMC', 'LTO', 'NaNiCl']  # TODO: Make labels dynamic.
    ax.set_yticks(y_array)
    y_labels = [item.get_text() for item in ax.get_yticklabels()]
    for i in np.arange(0, len(y_array), 1):
        y_labels[i] = legend_labels[i]
    ax.set_yticklabels(y_labels)
    ax.set_aspect(aspect=0.5)

    # Modify plot title.
    if plot_type == 'simple_payback_time':
        title = 'Simple payback time in years'
    elif plot_type == 'discounted_payback_time':
        title = 'Discounted payback time in years'
    elif plot_type == 'storage_capacity':
        title = 'Storage size in kWh'
    elif plot_type == 'efficiency':
        title = 'Efficiency in %%'
    elif plot_type == 'operation_cost_savings_annual':
        title = 'Annual operation cost savings in SGD'
    elif plot_type == 'operation_cost_savings_annual_percentage':
        title = 'Relative annual operation cost savings in %%'
    ax.title.set_text(title)

    # Save plot to SVG.
    fig.savefig(os.path.join(results_path, plot_type + '.svg'))


def plot_battery_cases_payback_comparison_lines(
        read_path,
        results_path,
        plot_type  # Choices: 'simple_payback_time', 'discounted_payback_time'.
):
    """Plot battery cases payback time comparison."""

    # Define plot settings.
    seaborn.set()
    plt.rcParams['font.serif'] = 'Arial'
    plt.rcParams['font.family'] = 'serif'

    # Load result data for plotting.
    results = pd.read_csv(read_path, index_col='battery_technology')
    years = results.columns
    x_array = np.arange(1, years.shape[0]+1, 1)
    techs = results.index
    colors = matplotlib.cm.Paired(np.linspace(0, 1, techs.shape[0]))

    # Create plot.
    fig, ax = plt.subplots(1, 1)
    for i in np.arange(0, techs.shape[0], 1):
        ax.scatter(
            x_array,
            np.array(results.iloc[i, :]),
            marker='o', facecolors='none', edgecolors=colors[i], s=70
        )
        ax.plot(
            x_array,
            np.array(results.iloc[i, :]),
            linestyle='-', color=colors[i], label='%s' % techs[i]
        )

    # Modify plot.
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylabel('years')
    ax.set_xticks(x_array)
    labels = [item.get_text() for item in ax.get_xticklabels()]
    for y in np.arange(0, len(x_array), 1):
        labels[y] = years[y]
    ax.set_xticklabels(labels)

    # Modify plot title.
    if plot_type == 'discounted_payback_time':
        title = 'Discounted payback time in years.'
    else:
        title = 'Simple payback time in years.'
    fig.suptitle(title)

    # Save plot to SVG.
    fig.savefig(os.path.join(results_path, plot_type + '_comparison.svg'))
