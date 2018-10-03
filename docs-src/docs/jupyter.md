# Jupyter environment
The Jupyter notebooks environment allows us to plan our compression session and load Distiller data summaries to study and analyze compression results.

Each notebook has embedded instructions and explanations, so here we provide only a brief description of each notebook.

## Installation
Jupyter and its dependencies are included as part of the main ```requirements.txt``` file, so there is no need for a dedicated installation step.<br>
However, to use the ipywidgets extension, you will need to enable it:
```
$ jupyter nbextension enable --py widgetsnbextension --sys-prefix
```

You may want to refer to the [ipywidgets extension installation documentation](http://ipywidgets.readthedocs.io/en/latest/user_install.html).

Another extension which requires special installation handling is [Qgrid](https://github.com/quantopian/qgrid).  Qgrid is a Jupyter notebook widget that adds interactive features, such as sorting, to Panadas DataFrames rendering.  To enable Qgrid:

```
$ jupyter nbextension enable --py --sys-prefix qgrid
```

## Launching the Jupyter server
There are all kinds of options to use when launching Jupyter which you can use.  The example below tells the server to listen to connections from any IP address, and not to launch the browser window, but of course, you are free to launch Jupyter any way you want.<br>
Consult the [user's guide](http://jupyter.readthedocs.io/en/latest/running.html) for more details.
```
$ jupyter-notebook --ip=0.0.0.0 --no-browser
```

## Using the Distiller notebooks
The Distiller Jupyter notebooks are located in the ```distiller/jupyter``` directory.<br>
They are provided as tools that you can use to prepare your compression experiments and study their results.
We welcome new ideas and implementations of Jupyter.

Roughly, the notebooks can be divided into three categories.
### Theory
- [jupyter/L1-regularization.ipynb](https://github.com/NervanaSystems/distiller/blob/master/jupyter/L1-regularization.ipynb): Experience hands-on how L1 and L2 regularization affect the solution of a toy loss-minimization problem, to get a better grasp on the interaction between regularization and sparsity.
- [jupyter/alexnet_insights.ipynb](https://github.com/NervanaSystems/distiller/blob/master/jupyter/alexnet_insights.ipynb): This notebook reviews and compares a couple of pruning sessions on Alexnet.  We compare distributions, performance, statistics and show some visualizations of the weights tensors.
### Preparation for compression
- [jupyter/model_summary.ipynb](https://github.com/NervanaSystems/distiller/blob/master/jupyter/model_summary.ipynb): Begin by getting familiar with your model.  Examine the sizes and properties of layers and connections.  Study which layers are compute-bound, and which are bandwidth-bound, and decide how to prune or regularize the model.
- [jupyter/sensitivity_analysis.ipynb](https://github.com/NervanaSystems/distiller/blob/master/jupyter/sensitivity_analysis.ipynb): If you performed pruning sensitivity analysis on your model, this notebook can help you load the results and graphically study how the layers behave.
- [jupyter/interactive_lr_scheduler.ipynb](https://github.com/NervanaSystems/distiller/blob/master/jupyter/interactive_lr_scheduler.ipynb): The learning rate decay policy affects pruning results, perhaps as much as it affects training results.  Graph a few LR-decay policies to see how they behave.
- [jupyter/jupyter/agp_schedule.ipynb](https://github.com/NervanaSystems/distiller/blob/master/jupyter/agp_schedule.ipynb): If you are using the Automated Gradual Pruner, this notebook can help you tune the schedule.
### Reviewing experiment results
- [jupyter/compare_executions.ipynb](https://github.com/NervanaSystems/distiller/blob/master/jupyter/compare_executions.ipynb): This is a simple notebook to help you graphically compare the results of executions of several experiments.
- [jupyter/compression_insights.ipynb](https://github.com/NervanaSystems/distiller/blob/master/jupyter/compression_insights.ipynb): This notebook is packed with code, tables and graphs to us understand the results of a compression session.  Distiller provides *summaries*, which are Pandas dataframes, which contain statistical information about you model.  We chose to use Pandas dataframes because they can be sliced, queried, summarized and graphed with a few lines of code.
