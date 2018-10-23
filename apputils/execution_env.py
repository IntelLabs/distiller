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

import sys
import os
import time
import platform
import logging
import logging.config
import numpy as np
import torch
from git import Repo, InvalidGitRepositoryError
HAVE_LSB = True
try:
    import lsb_release
except ImportError:
    HAVE_LSB = False

logger = logging.getLogger("app_cfg")

def log_execution_env_state(app_args, gitroot='.'):
    """Log information about the execution environment.

    It is recommeneded to log this information so it can be used for referencing
    at a later time.

    Args:
        app_args (dict): the command line arguments passed to the application
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
    logger.debug("App args: %s", app_args)


def config_pylogger(log_cfg_file, experiment_name, output_dir='logs'):
    """Configure the Python logger.

    For each execution of the application, we'd like to create a unique log directory.
    By default this library is named using the date and time of day, to that directories
    can be sorted by recency.  You can also name yor experiments and prefix the log
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
    return msglogger
