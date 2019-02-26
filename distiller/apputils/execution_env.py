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

import contextlib
import logging
import logging.config
import os
import platform
import shutil
import sys
import time

from git import Repo, InvalidGitRepositoryError
import numpy as np
import torch
try:
    import lsb_release
    HAVE_LSB = True
except ImportError:
    HAVE_LSB = False

logger = logging.getLogger("app_cfg")


def log_execution_env_state(config_filename=None, logdir=None, gitroot='.'):
    """Log information about the execution environment.

    It is recommeneded to log this information so it can be used for referencing
    at a later time.

    Args:
        config_filename: filename to store in logdir, if logdir is set
        logdir: log directory
        git_root: the path to the .git root directory
    """

    def log_git_state():
        """Log the state of the git repository.

        It is useful to know what git tag we're using, and if we have outstanding code.
        """
        try:
            repo = Repo(gitroot)
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

    logger.debug("Number of CPUs: %d", len(os.sched_getaffinity(0)))
    logger.debug("Number of GPUs: %d", torch.cuda.device_count())
    logger.debug("CUDA version: %s", torch.version.cuda)
    logger.debug("CUDNN version: %s", torch.backends.cudnn.version())
    logger.debug("Kernel: %s", platform.release())
    if HAVE_LSB:
        logger.debug("OS: %s", lsb_release.get_lsb_information()['DESCRIPTION'])
    logger.debug("Python: %s", sys.version)
    logger.debug("PyTorch: %s", torch.__version__)
    logger.debug("Numpy: %s", np.__version__)
    log_git_state()
    logger.debug("Command line: %s", " ".join(sys.argv))

    if (logdir is None) or (config_filename is None):
        return
    # clone configuration files to output directory
    configs_dest = os.path.join(logdir, 'configs')
    with contextlib.suppress(FileExistsError):
        os.makedirs(configs_dest)
    if os.path.exists(os.path.join(configs_dest, config_filename)):
        logger.debug('{} already exists in logdir'.format(
            os.path.basename(config_filename) or config_filename))
    else:
        shutil.copy(config_filename, configs_dest)


def config_pylogger(log_cfg_file, experiment_name, output_dir='logs'):
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
    msglogger = logging.getLogger()
    msglogger.logdir = logdir
    msglogger.log_filename = log_filename
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
