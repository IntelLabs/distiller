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
                    ax.legend(['mean'] + [str(i+1) for i in range(len(dfs_top1))])
                else:
                    ax.legend([str(i+1) for i in range(len(dfs_top1))])
                ax.set(xlabel="Episode", ylabel="(%)")            
                ax.fill_between(range(len(top1_min)), top1_max, top1_min, color="b", alpha=0.3)

                left_axs.append((ax, "Top1"))
            else:
                assert plot_type == 'compare'
                dfs_top1 = [df['smooth_top1'] for df in dfs]
                for p in dfs_top1:
                    #ax = p[start:end].plot(**plot_kwargs)
                    ax = p[start:].plot(**plot_kwargs)
                ax.legend([str(i+1) for i in range(len(dfs_top1))])
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