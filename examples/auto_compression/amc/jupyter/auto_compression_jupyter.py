import os
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
from scipy.stats.stats import pearsonr


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
    #ax.set_xticks(range(len(net_performance)), list(net_performance.keys()))


def plot_action_history(df, idx, action_type='action_history', ax=None, color=None):
    if ax is None:
        plt.figure()
        ax = plt
    
    record = df.iloc[idx]
    layer_sparsities = json.loads(record[action_type])
    #layer_sparsities = record[action_type]
    #layer_sparsities = layer_sparsities[1:-1].split(",")
    layer_densities = [1.- float(sparsity) for sparsity in layer_sparsities]
    ax.bar(range(len(layer_densities)), layer_densities, color=color)
    ax.set_title("Ep:{} - Top1:{:.1f}%\nMACs:{:.1f}%".format(record['episode'], 
                                                             record['top1'], 
                                                             record['normalized_macs']))


def smooth(data, win_size):
    if not win_size:
        return data
    win_size = max(0, win_size)
    return [np.mean(data[max(0, i-win_size):i]) for i in range(len(data))]


def plot_performance(title, dfs, alpha, window_size, top1, macs, params, reward, start=0, end=-1, plot_type='error'):
    plot_kwargs = {"figsize":(15,7), "lw": 1, "alpha": alpha, "title": title, "grid": True}
    smooth_kwargs = {"lw": 2 if window_size > 0 else 1, "legend": True, "grid": True}
    
    if not isinstance(dfs, list):
        dfs = [dfs]
    # Apply zoom
    df_end = min([len(df) for df in dfs])
    if end > 0:
        #df_end = min(df_end, end)
        end = min(df_end, end)
    else:
        end = df_end
    print(end)
    #dfs = [df[:df_end].copy() for df in dfs] 
    dfs = [df for df in dfs] 
    df = dfs[0]
    left_axs, right_axs = [], []
    
    if macs:
        ax = df['normalized_macs'][start:end].plot(**plot_kwargs, color="r")
        left_axs.append((ax, "MACs"))
        #ax.set(xlabel="Episode", ylabel="(%)")
        #ax.set_ylim([0,100])
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
                    ax.legend(['mean'] + [str(i) for i in range(len(dfs_top1))])
                else:
                    ax.legend([str(i) for i in range(len(dfs_top1))])
                ax.set(xlabel="Episode", ylabel="(%)")            
                ax.fill_between(range(len(top1_min)), top1_max, top1_min, color="b", alpha=0.3)

                left_axs.append((ax, "Top1"))
            else:
                assert plot_type == 'compare'
                dfs_top1 = [df['smooth_top1'] for df in dfs]
                for p in dfs_top1:
                    #ax = p[start:end].plot(**plot_kwargs)
                    ax = p[start:].plot(**plot_kwargs)
                ax.legend([str(i) for i in range(len(dfs_top1))])
                left_axs.append((ax, "Top1"))
        else:
            ax = df['top1'][start:end].plot(**plot_kwargs, color="b")
            left_axs.append((ax, "Top1"))
            #ax.set(xlabel="Episode", ylabel="(%)")
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
    #ax.set_ylim([0,100])
    #ax.grid(True, which='minor', axis='x', alpha=0.3)
    plt.xlabel("Episode")
    
    # The left axis might contain multiple ylabels
    if left_axs:
        # Pick an arbitrary axis 
        ax = left_axs[0][0]
        # Collect all of the labels
        ylabel = " / ".join([ax[1] for ax in left_axs]) #left_axs[0][1]
        ax.set(ylabel=ylabel)


def plot_2d_embeddings(top1, normalized_macs):
    plt.figure(figsize=(15,7))        
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
    color_grad = [str(1-i/len(top1)) for i in range(len(top1))]
    plt.scatter(normalized_macs, top1, color=color_grad, s=80, edgecolors='gray');

    
INTERVAL = 30 # Animation speed
WINDOW = 20

font = {'family': 'serif',
        'color':  'darkred',
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
        self.fig, self.ax = plt.subplots(figsize=(15,7))
        # Then setup FuncAnimation.
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=INTERVAL,
                                           frames=self.numpoints-2, 
                                           init_func=self.setup_plot, blit=True)

    def setup_plot(self):
        """Initialize drawing of the scatter plot."""
        x, y, s, c = next(self.stream)
        #self.annot = self.ax.annotate("txt", (10, 10))
        self.scat = self.ax.scatter(x, y, c=c, s=s, animated=False)
        self.scat.set_edgecolors('gray')
        self.scat.set_cmap('gray')
        self.width = max(self.xdata) - min(self.xdata) + 4
        self.height = max(self.ydata) - min(self.ydata) + 4
        self.ax.axis([min(self.xdata)-2, max(self.xdata)+2, 
                      min(self.ydata)-2, max(self.ydata)+2])
        
        self.annot = self.ax.text(min(self.xdata) + self.width/2, self.height/2, 
                     "", fontdict=font)
        # For FuncAnimation's sake, we need to return the artist we'll be using
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat, 

    def data_stream(self):
        numpoints = 0#len(self.xdata)
        colors = []
        xxx = 0
        while True:
            numpoints += 1
            win_len = min(WINDOW, numpoints)
            data = np.ndarray((4, win_len))
            start = max(0,numpoints-WINDOW-1)
            data[0, :] = self.xdata[start:start+win_len]
            data[1, :] = self.ydata[start:start+win_len]
            data[2, :] = [70] * win_len  # point size
            #data[3, :] = [np.random.random() for p in range(numpoints)]  # color
            # The color of the points is a gradient with larger values for "younger" points.
            # At each new frame we show one more point, and "age" each existing point by incrementaly  
            # reducing its color gradient.
            data[3, :] = [(1-i/(win_len+1)) for i in range(win_len)] 
            yield data

    def update(self, i):      
        """Update the scatter plot."""
        data = next(self.stream)
        self.annot.set_text(i)
        i = i % len(data)
            
        # Set x and y data
        xy = [(data[0,i], data[1,i]) for i in range(len(data[0,:]))]
        self.scat.set_offsets(xy)
        
        # Set colors
        self.scat.set_array(data[3])
        
        # We need to return the updated artist for FuncAnimation to draw..
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat, self.annot

    def show(self):
        plt.show()


def get_immediate_subdirs(a_dir):
    subdirs = [os.path.join(a_dir, name) for name in os.listdir(a_dir)
                    if os.path.isdir(os.path.join(a_dir, name)) and name != "ft"]
    subdirs.sort()
    return subdirs


def load_experiment_instances(dir_with_experiments):
    experiment_instance_dirs = get_immediate_subdirs(dir_with_experiments)
    return [pd.read_csv(os.path.join(dirname, "amc.csv"))
                        for dirname in experiment_instance_dirs]


def plot_experiment_comparison(df_list):
    df_len = min([len(df) for df in df_list])
    @interact(window_size=(0,50,5), top1=True, macs=True, params=False, reward=True, zoom=(0,df_len,1))
    def _plot_experiment_comparison(window_size=10, zoom=0):
        start = 0
        end = zoom if zoom > 0 else 0
        plot_performance("Compare AMC Experiment Executions (Top1)", df_list,
                         0.15, window_size, True, False, False, False, start, end, plot_type='compare')


def shorten_dir_name(df):
    df_grouped = df.groupby('dir')
    df['dir'] = df['dir'].map(lambda x: x[len("experiments/plain20-ddpg-private/"):])
    df['exp'] = df['dir'].map(lambda x: x.split("___")[0][-1])


def create_fig(title, xlabel, ylabel):
    plt.figure(figsize=(15, 7))
    plt.grid(True)
    # plt.legend()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


def plot_networks(df, edgecolors):
    size = df.macs / max(df.macs) * 100
    plt.scatter(df.top1, df.search_top1, s=size, c=df.index, edgecolors=edgecolors, cmap='gray')
    print("Best network: %.2f" % max(df.top1))
    print("Pearson: %.3f" % pearsonr(df.top1, df.search_top1)[0])
    df_sorted = df.sort_values(by=['top1'], inplace=False, ascending=False)
    # Five best
    print(df_sorted[:5][['exp', 'search_top1', 'top1', 'name']])  # [('name','top1')])


def plot_networks_by_experiment(df, edgecolors, create_figure=True):
    # Group by experiment directory
    df_grouped = df.groupby('exp')
    size = df.macs / max(df.macs) * 100
    legend = []
    for exp_dir, df_experiment in df_grouped:
        a = plt.scatter(df_experiment.top1, df_experiment.search_top1, s=size,
                        edgecolors=edgecolors, alpha=0.5)
        p = pearsonr(df_experiment.top1, df_experiment.search_top1)[0]
        legend.append((a, "%s: Best Top1=%.2f  Pearson: %.3f" % (exp_dir, max(df_experiment.top1), p)))
    plots, labels = zip(*legend)
    plt.legend(plots, labels)