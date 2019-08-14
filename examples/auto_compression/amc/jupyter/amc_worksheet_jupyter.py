import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import csv
from matplotlib.ticker import FuncFormatter
import ipywidgets as widgets
from ipywidgets import interactive, interact, Layout
import matplotlib.pylab as pylab
import matplotlib.animation as animation
from matplotlib import animation, rc

params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 7),
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'xx-large',
          'xtick.labelsize': 'x-large',
          'ytick.labelsize': 'x-large'}
pylab.rcParams.update(params)


# plt.style.use('seaborn') # pretty matplotlib plots


def to_percent(y, position):
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    if y < 1:
        y = 100 * y
    s = "{:.1f}".format(y)

    # The percent symbol needs escaping in latex
    if matplotlib.rcParams['text.usetex'] is True:
        return s + r'$\%$'
    else:
        return s + '%'


# Widen the cells to get entire rows in the screen.
from IPython.core.display import display, HTML
# display(HTML("<style>.container { width:100% !important; }</style>"))

import json


def plot_layer_compute_densities(df, idx, ax=None, color=None):
    if ax is None:
        plt.figure()
        ax = plt

    record = df.iloc[idx]
    net_performance = json.loads(record["performance"])
    ax.bar(range(len(net_performance)), list(net_performance.values()), color=color, align='center')
    ax.set_title("Ep:{} - Top1:{:.1f}%\nMACs:{:.1f}%".format(record['episode'],
                                                             record['top1'],
                                                             record['normalized_macs']))
    # ax.set_xticks(range(len(net_performance)), list(net_performance.keys()))


def plot_action_history(df, idx, action_type='action_history', ax=None, color=None):
    if ax is None:
        plt.figure()
        ax = plt

    record = df.iloc[idx]
    layer_sparsities = json.loads(record[action_type])
    # layer_sparsities = record[action_type]
    # layer_sparsities = layer_sparsities[1:-1].split(",")
    layer_densities = [1. - float(sparsity) for sparsity in layer_sparsities]
    ax.bar(range(len(layer_densities)), layer_densities, color=color)
    ax.set_title("Ep:{} - Top1:{:.1f}%\nMACs:{:.1f}%".format(record['episode'],
                                                             record['top1'],
                                                             record['normalized_macs']))


def smooth(data, win_size):
    if not win_size:
        return data
    win_size = max(0, win_size)
    return [np.mean(data[max(0, i - win_size):i]) for i in range(len(data))]


def plot_performance(title, dfs, alpha, window_size, top1, macs, params, reward, start=0, end=-1, plot_type='error'):
    plot_kwargs = {"figsize": (15, 7), "lw": 1, "alpha": alpha, "title": title, "grid": True}
    smooth_kwargs = {"lw": 2 if window_size > 0 else 1, "legend": True, "grid": True}

    if not isinstance(dfs, list):
        dfs = [dfs]
    # Apply zoom
    df_end = min([len(df) for df in dfs])
    if end > 0:
        # df_end = min(df_end, end)
        end = min(df_end, end)
    else:
        end = df_end
    print(end)
    # dfs = [df[:df_end].copy() for df in dfs]
    dfs = [df for df in dfs]
    df = dfs[0]
    left_axs, right_axs = [], []

    if macs:
        ax = df['normalized_macs'][start:end].plot(**plot_kwargs, color="r")
        left_axs.append((ax, "MACs"))
        # ax.set(xlabel="Episode", ylabel="(%)")
        # ax.set_ylim([0,100])
        df['smooth_normalized_macs'] = smooth(df['normalized_macs'], window_size)
        df['smooth_normalized_macs'][start:end].plot(**smooth_kwargs, color="r")

    if top1:
        for df in dfs:
            df['smooth_top1'] = smooth(df['top1'], window_size)

        if len(dfs) > 1:
            plot_kwargs['alpha'] = 1.0
            plot_kwargs['legend'] = True
            if plot_type == 'error':
                top1_len = min([len(df) for df in dfs])
                dfs = [df[:top1_len] for df in dfs]
                dfs_top1 = [df['smooth_top1'] for df in dfs]
                dfs_top1_dp = pd.DataFrame(dfs_top1)
                top1_min = dfs_top1_dp.min(axis=0)
                top1_max = dfs_top1_dp.max(axis=0)
                top1_mean = dfs_top1_dp.mean(axis=0)

                display_mean = False
                if display_mean:
                    ax = top1_mean.plot(**plot_kwargs, color="b")

                for p in dfs_top1:
                    ax = p[start:end].plot(**plot_kwargs)
                if display_mean:
                    ax.legend(['mean'] + [str(i + 1) for i in range(len(dfs_top1))])
                else:
                    ax.legend([str(i + 1) for i in range(len(dfs_top1))])
                ax.set(xlabel="Episode", ylabel="(%)")
                ax.fill_between(range(len(top1_min)), top1_max, top1_min, color="b", alpha=0.3)

                left_axs.append((ax, "Top1"))
            else:
                assert plot_type == 'compare'
                dfs_top1 = [df['smooth_top1'] for df in dfs]
                for p in dfs_top1:
                    # ax = p[start:end].plot(**plot_kwargs)
                    ax = p[start:].plot(**plot_kwargs)
                ax.legend([str(i + 1) for i in range(len(dfs_top1))])
                left_axs.append((ax, "Top1"))
        else:
            ax = df['top1'][start:end].plot(**plot_kwargs, color="b")
            left_axs.append((ax, "Top1"))
            # ax.set(xlabel="Episode", ylabel="(%)")
            df['smooth_top1'][start:end].plot(**smooth_kwargs, color="b")

    if params:
        ax = df['normalized_nnz'][start:end].plot(**plot_kwargs, color="black")
        ax.set(xlabel="Episode", ylabel="(%)")
        df['smooth_normalized_nnz'] = smooth(df['normalized_nnz'], window_size)
        df['smooth_normalized_nnz'][start:end].plot(**smooth_kwargs, color="black")

    if reward:
        ax = df['reward'][start:end].plot(**plot_kwargs, secondary_y=True, color="g")
        ax.set(xlabel="Episode", ylabel="Reward")
        df['smooth_reward'] = smooth(df['reward'], window_size)
        df['smooth_reward'][start:end].plot(**smooth_kwargs, secondary_y=True, color="g")
        # ax.set_ylim([0,100])
    # ax.grid(True, which='minor', axis='x', alpha=0.3)
    plt.xlabel("Episode")

    # The left axis might contain multiple ylabels
    if left_axs:
        # Pick an arbitrary axis
        ax = left_axs[0][0]
        # Collect all of the labels
        ylabel = " / ".join([ax[1] for ax in left_axs])  # left_axs[0][1]
        ax.set(ylabel=ylabel)


def plot_2d_embeddings(top1, normalized_macs):
    plt.figure(figsize=(15, 7))
    plt.title('Projection of Discovered Networks ({})'.format(len(top1)))
    plt.xlabel('Normalized MACs')
    plt.ylabel('Top1 Accuracy')

    # Create the formatter using the function to_percent. This multiplies all the
    # default labels by 100, making them all percentages
    formatter = FuncFormatter(to_percent)

    # Set the formatter
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.gca().xaxis.set_major_formatter(formatter)

    # Use color gradients to show the "age" of the network:
    # Lighter networks were discovered earlier than darker ones.
    color_grad = [str(1 - i / len(top1)) for i in range(len(top1))]
    plt.scatter(normalized_macs, top1, color=color_grad, s=80, edgecolors='gray');


INTERVAL = 30  # Animation speed
WINDOW = 20

font = {'family': 'serif',
        'color': 'darkred',
        'weight': 'normal',
        'alpha': 0.50,
        'size': 32,
        }


# Based on these two helpful example code:
# https://stackoverflow.com/questions/9401658/how-to-animate-a-scatter-plot
# http://louistiao.me/posts/notebooks/embedding-matplotlib-animations-in-jupyter-notebooks/.
# Specifically, the use of IPython.display is missing from the first example, but most of the animation code
# leverages code from there.
class AnimatedScatter(object):
    """An animated scatter plot using matplotlib.animations.FuncAnimation."""

    def __init__(self, xdata, ydata):
        assert len(xdata) == len(ydata)
        self.numpoints = len(xdata)
        self.xdata = xdata
        self.ydata = ydata
        self.stream = self.data_stream()

        # Setup the figure and axes...
        self.fig, self.ax = plt.subplots(figsize=(15, 7))
        # Then setup FuncAnimation.
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=INTERVAL,
                                           frames=self.numpoints - 2,
                                           init_func=self.setup_plot, blit=True)

    def setup_plot(self):
        """Initialize drawing of the scatter plot."""
        x, y, s, c = next(self.stream)
        # self.annot = self.ax.annotate("txt", (10, 10))
        self.scat = self.ax.scatter(x, y, c=c, s=s, animated=False)
        self.scat.set_edgecolors('gray')
        self.scat.set_cmap('gray')
        self.width = max(self.xdata) - min(self.xdata) + 4
        self.height = max(self.ydata) - min(self.ydata) + 4
        self.ax.axis([min(self.xdata) - 2, max(self.xdata) + 2,
                      min(self.ydata) - 2, max(self.ydata) + 2])

        self.annot = self.ax.text(min(self.xdata) + self.width / 2, self.height / 2,
                                  "", fontdict=font)
        # For FuncAnimation's sake, we need to return the artist we'll be using
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat,

    def data_stream(self):
        numpoints = 0  # len(self.xdata)
        colors = []
        xxx = 0
        while True:
            numpoints += 1
            win_len = min(WINDOW, numpoints)
            data = np.ndarray((4, win_len))
            start = max(0, numpoints - WINDOW - 1)
            data[0, :] = self.xdata[start:start + win_len]
            data[1, :] = self.ydata[start:start + win_len]
            data[2, :] = [70] * win_len  # point size
            # data[3, :] = [np.random.random() for p in range(numpoints)]  # color
            # The color of the points is a gradient with larger values for "younger" points.
            # At each new frame we show one more point, and "age" each existing point by incrementaly
            # reducing its color gradient.
            data[3, :] = [(1 - i / (win_len + 1)) for i in range(win_len)]
            yield data

    def update(self, i):
        """Update the scatter plot."""
        data = next(self.stream)
        self.annot.set_text(i)
        i = i % len(data)

        # Set x and y data
        xy = [(data[0, i], data[1, i]) for i in range(len(data[0, :]))]
        self.scat.set_offsets(xy)

        # Set colors
        self.scat.set_array(data[3])

        # We need to return the updated artist for FuncAnimation to draw..
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat, self.annot

    def show(self):
        plt.show()


# PCA Projection

from sklearn.decomposition import PCA
from collections import OrderedDict

def net2nparray(record):
    net_layer_sizes = json.loads(record["performance"])
    net_as_nparray = np.array(list(net_layer_sizes.values()), dtype=np.float32)
    return net_as_nparray

def collect_info(df):
    arch_shapes_dict = OrderedDict()
    for i in range(len(df.index)):
        record = df.iloc[i]
        episode = record['episode']
        top1 = record['top1']
        arch_shapes_dict[episode] = (net2nparray(record), top1)
    return arch_shapes_dict

def pca_projection(arch_shapes_dict, title, show_legend, show_episode_ids):
    import math
    pca = PCA(n_components=2)
    # Unpack the dictionary
    packed_arch_shapes = arch_shapes_dict.values()
    arch_shapes = list(shape_vec for shape_vec, top1 in packed_arch_shapes)
    arch_top1s = list(top1 for shape_vec, top1 in packed_arch_shapes)
    principal_components = pca.fit_transform(arch_shapes)
    #episodes = [str(episode) for episode in arch_shapes_dict.keys()]
    pc_df = pd.DataFrame(data=principal_components, columns = ['pc1', 'pc2'])
    #pc_df = pd.concat([pc_df, pd.Series(episodes)], axis = 1)

    fig = plt.figure(figsize = (10,10))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title(title, fontsize=20)

    cnt_nets = len(arch_shapes_dict)
    colors = ['lightskyblue', 'orange', 'red', 'green', 'blue', 'purple', 'pink','lime']

    episodes = list(arch_shapes_dict.keys())
    #rescaled_top1 = [math.log(top1)*30 for top1 in arch_top1s]
    rescaled_top1 = arch_top1s * 30
    for i in range(cnt_nets):
        x,y = pc_df.pc1, pc_df.pc2
        ax.scatter(x[i], y[i], label=episodes[i], s=rescaled_top1[i],
                   color=colors[episodes[i]//100], alpha=0.5)
        if show_episode_ids:
            ax.text(x[i], y[i], episodes[i], fontsize=9)
    if show_legend:
        plt.legend(labels, loc='best')
    ax.grid()
    print("Which components explain the variance best", pca.explained_variance_ratio_)
