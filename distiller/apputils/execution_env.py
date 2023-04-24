#
# Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Log information regarding the execution environment.

This is helpful if you want to recreate an experiment at a later time, or if
you want to understand the environment in which you execute the training.
"""

import logging
import logging.config
import operator
import os
import platform
import shutil
import sys
import time
import pkg_resources

from git import Repo, InvalidGitRepositoryError
import numpy as np
import torch
try:
    import lsb_release
    HAVE_LSB = True
except ImportError:
    HAVE_LSB = False

logger = logging.getLogger("app_cfg")


def log_execution_env_state(config_paths=None, logdir=None):
    """Log information about the execution environment.

    Files in 'config_paths' will be copied to directory 'logdir'. A common use-case
    is passing the path to a (compression) schedule YAML file. Storing a copy
    of the schedule file, with the experiment logs, is useful in order to
    reproduce experiments.

    Args:
        config_paths: path(s) to config file(s), used only when logdir is set
        logdir: log directory
        git_root: the path to the .git root directory
    """

    def log_git_state():
        """Log the state of the git repository.

        It is useful to know what git tag we're using, and if we have outstanding code.
        """
        try:
            repo = Repo(os.path.join(os.path.dirname(__file__), '..', '..'))
            assert not repo.bare
        except InvalidGitRepositoryError:
            logger.debug("Cannot find a Git repository.  You probably downloaded an archive of Distiller.")
            return

        if repo.is_dirty():
            logger.debug("Git is dirty")
        try:
            branch_name = repo.active_branch.name
        except TypeError:
            branch_name = "None, Git is in 'detached HEAD' state"
        logger.debug("Active Git branch: %s", branch_name)
        logger.debug("Git commit: %s" % repo.head.commit.hexsha)

    try:
        num_cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        num_cpus = os.cpu_count()
    logger.debug("Number of CPUs: %d", num_cpus)
    logger.debug("Number of GPUs: %d", torch.cuda.device_count())
    logger.debug("CUDA version: %s", torch.version.cuda)
    logger.debug("CUDNN version: %s", torch.backends.cudnn.version())
    logger.debug("Kernel: %s", platform.release())
    if HAVE_LSB:
        logger.debug("OS: %s", lsb_release.get_lsb_information()['DESCRIPTION'])
    logger.debug("Python: %s", sys.version)
    try:
        logger.debug("PYTHONPATH: %s", os.environ['PYTHONPATH'])
    except KeyError:
        pass
    def _pip_freeze():
        return {x.key:x.version for x in sorted(pkg_resources.working_set,
                                                key=operator.attrgetter('key'))}
    logger.debug("pip freeze: {}".format(_pip_freeze()))
    log_git_state()
    logger.debug("Command line: %s", " ".join(sys.argv))

    if (logdir is None) or (config_paths is None):
        return

    # clone configuration files to output directory
    configs_dest = os.path.join(logdir, 'configs')

    if isinstance(config_paths, str) or not hasattr(config_paths, '__iter__'):
        config_paths = [config_paths]
    for cpath in config_paths:
        os.makedirs(configs_dest, exist_ok=True)

        if os.path.exists(os.path.join(configs_dest, os.path.basename(cpath))):
            logger.debug('{} already exists in logdir'.format(
                os.path.basename(cpath) or cpath))
        else:
            try:
                shutil.copy(cpath, configs_dest)
            except OSError as e:
                logger.debug('Failed to copy of config file: {}'.format(str(e)))


def config_pylogger(log_cfg_file, experiment_name, output_dir='logs', verbose=False):
    """Configure the Python logger.

    For each execution of the application, we'd like to create a unique log directory.
    By default this directory is named using the date and time of day, so that directories
    can be sorted by recency.  You can also name your experiments and prefix the log
    directory with this name.  This can be useful when accessing experiment data from
    TensorBoard, for example.
    """
    timestr = time.strftime("%Y.%m.%d-%H%M%S")
    exp_full_name = timestr if experiment_name is None else experiment_name + '___' + timestr
    logdir = os.path.join(output_dir, exp_full_name)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    log_filename = os.path.join(logdir, exp_full_name + '.log')
    if os.path.isfile(log_cfg_file):
        logging.config.fileConfig(log_cfg_file, defaults={'logfilename': log_filename})
    else:
        print("Could not find the logger configuration file (%s) - using default logger configuration" % log_cfg_file)
        apply_default_logger_cfg(log_filename)
    msglogger = logging.getLogger()
    msglogger.logdir = logdir
    msglogger.log_filename = log_filename
    if verbose:
        msglogger.setLevel(logging.DEBUG)
    msglogger.info('Log file for this run: ' + os.path.realpath(log_filename))

    # Create a symbollic link to the last log file created (for easier access)
    try:
        os.unlink("latest_log_file")
    except FileNotFoundError:
        pass
    try:
        os.unlink("latest_log_dir")
    except FileNotFoundError:
        pass
    try:
        os.symlink(logdir, "latest_log_dir")
        os.symlink(log_filename, "latest_log_file")
    except OSError:
        msglogger.debug("Failed to create symlinks to latest logs")
    return msglogger


def apply_default_logger_cfg(log_filename):
    d = {
        'version': 1,
        'formatters': {
            'simple': {
                'class': 'logging.Formatter',
                'format': '%(asctime)s - %(message)s'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': 'INFO',
            },
            'file': {
                'class': 'logging.FileHandler',
                'filename': log_filename,
                'mode': 'w',
                'formatter': 'simple',
            },
        },
        'loggers': {
            '': {  # root logger
                'level': 'DEBUG',
                'handlers': ['console', 'file'],
                'propagate': False
            },
            'app_cfg': {
                'level': 'DEBUG',
                'handlers': ['file'],
                'propagate': False
            },
        }
    }

    logging.config.dictConfig(d)
