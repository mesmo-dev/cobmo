"""Plot function definitions."""

import numpy as np
import pandas as pd
import matplotlib.cm
import matplotlib.pyplot as plt
import seaborn


def plot_battery_cases_bubbles(
        case,
        filepath_read,
        save_path,
        filename,
        labels,
        savepdf,
        pricing_method
):
    """

    :param case:
    :param payback_type:
    :param filepath_read:
    :param save_path:
    :param save_plots:
    :return:
    """
    seaborn.set()
    plt.rcParams['font.serif'] = "Arial"  # Palatino Linotype
    plt.rcParams['font.family'] = "serif"

    results = pd.read_csv(filepath_read, index_col='battery_technology')
    years = results.columns
    techs = results.index
    x_array = np.arange(1, years.shape[0]+1, 1)
    y_array = np.arange(1, techs.shape[0]+1, 1)
    colors = matplotlib.cm.Paired(np.linspace(0, 1, techs.shape[0]))
    # >> Many color maps here:
    # https://matplotlib.org/2.0.2/examples/color/colormaps_reference.html

    fig2, all_techs = plt.subplots(1, 1)

    for t in np.arange(0, techs.shape[0], 1):
        for y in np.arange(0, len(x_array), 1):
            all_techs.scatter(
                x_array[y],
                y_array[t],
                marker='o', facecolors=colors[t], edgecolors='none',
                s=(2000.0/max(results.max(axis=0)) * results.iloc[t, y] if max(results.max(axis=0)) != 0.0 else 0.0),
                # s=results.iloc[t, y]*0.5 if labels == 'savings_year' else results.iloc[t, y],
                alpha=0.7,
                # label='%s' % techs[t]
            )

            if results.iloc[t, y] != 0.0:
                all_techs.text(
                    x_array[y], y_array[t],
                    (
                        format(results.iloc[t, y], ('.0f' if
                                                    labels != 'efficiency'
                                                    and labels != 'savings_year_percentage'
                                                    else '.2f'))
                        + ('%' if labels == 'savings_year_percentage' else '')
                     ),
                    # style='italic',
                    weight='bold',
                    fontsize=9,
                    bbox={'facecolor': 'none', 'alpha': 0.5, 'pad': 1, 'edgecolor': 'none'},
                    ha='center', va='center'
                )

    # all_techs.legend(loc='upper right', fontsize=9)  # TODO: add labels per tech
    # potential solution for adding labels properly:
    # https://stackoverflow.com/questions/13588920/stop-matplotlib-repeating-labels-in-legend

    # circle_x = x_array[0]
    # circle_y = (y_array[0] + y_array[-1])/2
    # circle = plt.Circle((circle_x, circle_y), radius=1)
    # all_techs.add_patch(circle)
    # all_techs.text(
    #     circle_x, circle_x,
    #     '%s' % labels,
    #     # style='italic',
    #     weight='bold',
    #     fontsize=15,
    #     bbox={'facecolor': 'none', 'alpha': 0.5, 'pad': 1, 'edgecolor': 'none'},
    #     ha='center', va='center'
    # )

    # Changing names in the x axis
    all_techs.set_xticks(x_array)
    x_labels = [item.get_text() for item in all_techs.get_xticklabels()]
    for y in np.arange(0, len(x_array), 1):
        x_labels[y] = years[y]
    all_techs.set_xticklabels(x_labels)

    legend_labels = ['Flooded LA', 'VRLA', 'LFP', 'NCA', 'NMC', 'LTO', 'NaNiCl']  # , 'NaNiCl'
    all_techs.set_yticks(y_array)
    y_labels = [item.get_text() for item in all_techs.get_yticklabels()]
    for i in np.arange(0, len(y_array), 1):
        y_labels[i] = legend_labels[i]
    all_techs.set_yticklabels(y_labels)

    # Title and saving
    if labels == 'savings_year':
        title = 'Savings per year — ' + pricing_method + ' — Case: %s — bubbles: SGD/year' % case

    if labels == 'savings_year_percentage':
        title = 'Savings per year as share — ' + pricing_method + ' — Case: %s — bubbles: %%' % case

    elif labels == 'storage_size':
        title = 'Storage size — ' + pricing_method + ' — Case: %s — bubbles: kWh' % case

    elif labels == 'simple_payback':
        title = 'Simple Payback — ' + pricing_method + ' — Case: %s — bubbles: years' % case

    elif labels == 'discounted_payback':
        title = 'Discounted Payback — ' + pricing_method + ' — Case: %s — bubbles: years' % case

    elif labels == 'efficiency':
        title = 'Efficiency — ' + pricing_method + ' — Case: %s — bubbles: [-]' % case

    elif labels == 'investment':
        title = 'Investment — ' + pricing_method + ' — Case: %s — bubbles: SGD/kWh' % case

    all_techs.title.set_text(title)

    all_techs.set_aspect(aspect=0.5)
    plt.show()

    fig2.savefig(save_path + '/' + filename + '.svg', format='svg', dpi=1200)
    if savepdf == 1:
        fig2.savefig(save_path + '/' + filename + '.pdf')


def plot_battery_cases(
        case,
        payback_type,
        filepath_read,
        save_path,
        save_plots='summary'  # 'summary + each'
):
    """

    :param case:
    :param payback_type:
    :param filepath_read:
    :param save_path:
    :param save_plots:
    :return:
    """
    seaborn.set()
    plt.rcParams['font.serif'] = "Palatino Linotype"
    plt.rcParams['font.family'] = "serif"

    results = pd.read_csv(filepath_read, index_col='battery_technology')
    years = results.columns
    x_array = np.arange(1, years.shape[0]+1, 1)
    techs = results.index
    colors = matplotlib.cm.Paired(np.linspace(0, 1, techs.shape[0]))
    # >> Many color maps here:
    # https://matplotlib.org/2.0.2/examples/color/colormaps_reference.html

    fig2, all_techs = plt.subplots(1, 1)

    for i in np.arange(0, techs.shape[0], 1):
        if save_plots == 'each':
            fig, one_tech = plt.subplots(1, 1)
            one_tech.scatter(
                x_array,
                np.array(results.iloc[i, :]),
                marker='o', facecolors='none', edgecolors=colors[i], s=70  # '#0074BD'
            )
            one_tech.plot(
                x_array,
                np.array(results.iloc[i, :]),
                linestyle='-', color=colors[i],
                label='case: %s | payback: %s' % (case, payback_type)
            )
            one_tech.set_ylabel('Payback year')
            # one_tech.set_xlabel('year')
            fig.legend(loc='upper right', fontsize=9)

            one_tech.grid(True, which='both')
            one_tech.grid(which='minor', alpha=0.5)

            title = 'Technology: %s' % techs[i]
            fig.suptitle(title)

            filename = case + '_case-' + payback_type + '_payback-' + techs[i]
            fig.savefig(save_path + filename + '.svg', format='svg', dpi=1200)

            # Filling in the global plot
            all_techs.scatter(
                x_array,
                np.array(results.iloc[i, :]),
                marker='o', facecolors='none', edgecolors=colors[i], s=70  # facecolors='none', edgecolors='#0074BD',
            )
            all_techs.plot(
                x_array,
                np.array(results.iloc[i, :]),
                linestyle='-', color=colors[i], label='%s' % techs[i]  # color='#0074BD',
            )

        elif save_plots == 'summary':
            # Filling in the global plot
            all_techs.scatter(
                x_array,
                np.array(results.iloc[i, :]),
                marker='o', facecolors='none', edgecolors=colors[i], s=70  # facecolors='none', edgecolors='#0074BD',
            )
            all_techs.plot(
                x_array,
                np.array(results.iloc[i, :]),
                linestyle='-', color=colors[i], label='%s' % techs[i]  # color='#0074BD',
            )

    all_techs.legend(loc='upper right', fontsize=9)
    all_techs.set_ylabel('years')

    # Changing names in the x axis
    all_techs.set_xticks(x_array)
    labels = [item.get_text() for item in all_techs.get_xticklabels()]
    for y in np.arange(0, len(x_array), 1):
        labels[y] = years[y]
    all_techs.set_xticklabels(labels)

    # Title and saving
    title = 'Case: %s  |  Payback:  %s' % (case, payback_type)
    fig2.suptitle(title)
    plt.show()

    filename2 = case + '_case-' + payback_type + '_payback-all_techs'
    fig2.savefig(save_path + filename2 + '.svg', format='svg', dpi=1200)