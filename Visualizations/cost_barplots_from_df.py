import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import create_df

""" Documentation
- Variables used throughout the functions:
    - input: String that indicates the type of plot to produce. Current choices are:
        * "worst" (Display one possible worst-case scenario where an algorithm incurs maximal cost in rounds where it 
                   is considered by MIN^det and minimal cost in the remaining rounds. 
                   It still incurs some (arbitrarily set) cost in those remaining rounds.)
        * "equal" (All algorithms always incur exactly the same cost.)
        * "upper-bound-worst" (An exaggeration of "worst" where the algorithms incur cost 0 in rounds 
                               when they are not considered by MIN^det)
    - phi: Upper bound on the cost in each round.
    - m: Number of algorithms.
    - n_cycles: Number of cycles.
    - with_title: Whether to display the title of the plot or not (Boolean)
    - with_min_det: Whether to display an additional bar that shows the cost of MIN^det throughout the rounds.
- Dataframe: Shape = m x n_cycles
- The plots only display the cost incurred by the algorithms used by MIN^det. They do not show the cost incurred by
  MIN^det for switching from one algorithm to the next. 
"""


def get_df(phi, m, n_cycles, input, with_title, with_min_det):
    """
    Wrapper function for the creation of a pandas dataframe that shows the costs incurred by the
    algorithms A_0, ..., A_{m-1} during the cycles.
    """
    if with_min_det:  # TODO: implement another bar for the cost of MIN^det
        if input == "worst":
            df = create_df.create_df_worst_with_min_det(phi, m, n_cycles)
        elif input == "equal":
            df = create_df.create_df_equal_with_min_det(phi, m, n_cycles)
        elif input == "upper-bound-worst":
            df = create_df.upper_bound_worst_case_with_min_det(phi, m, n_cycles)
    else:
        if input == "worst":
            df = create_df.create_df_worst(phi, m, n_cycles)
        elif input == "equal":
            df = create_df.create_df_equal(phi, m, n_cycles)
        elif input == "upper-bound-worst":
            df = create_df.upper_bound_worst_case(phi, m, n_cycles)
    return df


def label_columns(df):
    """
    Relabel the columns (representing the cycles) of the Dataframe in a more meaningful way: "0" -> "Cycle 0" etc.
    """
    n_cols = df.shape[1]
    names = []
    for i in range(n_cols):
        names.append("Cycle "+str(i))
    df.columns = names


def label_rows(df):
    """
    Relabel the rows (representing the algorithms) of the Dataframe in a more meaningful way: "0" -> "A_0"
    """
    df.rename(lambda i: f"$A_{str(i)}$", axis='index', inplace=True)


def label_df(df, with_min_det):
    """
    Wrapper function for the functions label_columns and label_rows. Applies both function to a given Dataframe.
    """
    label_columns(df)
    label_rows(df)
    if with_min_det:
        df.rename(mapper={f'$A_{df.shape[0]-1}$': '$MIN^{det}$'}, inplace=True, errors='raise')


def set_x_axis(ax, m, with_min_det):
    """
    Set the tick labels of the x-axis of the plot to the name of the algorithm of the corresponding bar.
    :param ax: ax object (Matplotlib subplot)
    """
    ticks = list(range(m + int(with_min_det)))
    tick_labels = [f"$A_{i}$" for i in range(m)]
    if with_min_det:
        tick_labels += ["$MIN^{det}$"]
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels)


def min_det_bound(m, n_cycles):
    """
    Computes the accumulated cost of the algorithm so far in the last m cycles. The output consists of two lists that
    can be used to plot the line representing the accumulated costs of MIN^det at the times when the corresponding
    algorithm was used by MIN^det the last time. The first list contains the x-coordinates of the important points of
    the line, the second list contains the y-coordinates. The first and the last entries of both lists are used to
    extend the line at its ends so that it does not end in the middle of the bars.
    :return: 2 lists of the same size, can be used to plot a line.
    """
    x = [-0.5] + list(range(m)) + [m - 0.5]
    y_iterator = [n_cycles - m + 1] + list(range(n_cycles - m + 1, n_cycles + 1)) + [n_cycles]
    y = [(n_cycles - m + 1) + (i % m) for i in y_iterator]

    return x, y


def accumulated_costs(df, t):
    """
    Computes the accumulated costs of the algorithms in a given cycle.
    :param t: Cycle number
    :return: Array of length m with the accumulated costs of A_0, ..., A_{m-1} in cycle t.
    """
    acc_costs = np.zeros(len(df['Cycle 0']))
    for i in range(t+1):
        acc_costs += np.array(df[f'Cycle {i}'].to_list())

    return acc_costs


def dash_min_det_areas(bars, m, with_min_det):
    """
    Dash the areas of the bars that represent cost incurred while being considered by the MIN^det algorithm.
    :param bars: Collection (tuple) of matplotlib bar objects (matplotlib.patches.Rectangle).
    """
    bar_count = 0
    alg_count = 0
    for bar in bars:
        if bar_count % m == alg_count % m:
            bar.set_hatch('///')
        bar_count += 1
        if bar_count % (m+int(with_min_det)) == 0:
            alg_count += 1
            if with_min_det:
                bar.set_hatch('///')
                alg_count += 1


def plot_cost_bars(phi, m, n_cycles, input, with_title, with_min_det):
    """
    Overall wrapper function for the plot creation.
    """
    df = get_df(phi, m, n_cycles, input, with_title, with_min_det)
    label_df(df, with_min_det)

    if with_min_det:
        n_bars = m+1
    else:
        n_bars = m

    plt.close("all")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    bars = ax.bar(range(n_bars), df['Cycle 0'], edgecolor='black', label="Cycle 0", width=0.85)
    for i in range(1, n_cycles):
        bars += ax.bar(range(n_bars), df[f'Cycle {i}'], bottom=accumulated_costs(df, i - 1), edgecolor='black',
                       label=f"Cycle {i}", width=0.85)
    set_x_axis(ax, m, with_min_det)
    ax.legend(fontsize='small', ncol=2)
    # ax.legend(fontsize='x-small', ncol=2, bbox_to_anchor=(1.1, 1.05))
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
    #          ncol=4, fancybox=True, shadow=True, fontsize='x-small')

    dash_min_det_areas(bars, m, with_min_det)

    if with_min_det:
        ax.axvline(x=2.505, linestyle='--', color='black')

    if with_title:
        plt.title("Accumulated costs during the different cycles.")

    plt.xlabel("Algorithm")
    plt.ylabel(r"Accumulated costs in units of $\gamma\varphi$")
    if with_min_det:
        if with_title:
            plt.savefig('Plots/' + input + '_with_min_det_' + 'with_title_' + str(m) + 'x' + str(n_cycles) + '.png')
        else:
            plt.savefig('Plots/' + input + '_with_min_det_' + str(m) + 'x' + str(n_cycles) + '.png')
    else:
        if with_title:
            plt.savefig('Plots/' + input + '_with_title_' + str(m) + 'x' + str(n_cycles) + '.png')
        else:
            plt.savefig('Plots/' + input + '_' + str(m) + 'x' + str(n_cycles) + '.png')
    plt.show()
