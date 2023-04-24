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

"""
$ python3 amc.py --arch=resnet20_cifar ${CIFAR10_PATH} --resume=../../ssl/checkpoints/checkpoint_trained_dense.pth.tar --amc --amc-procol=mac-constrained --amc-action-range 0.05 1.0 --amc-target-density=0.5 -p=50 --etes=0.075 --amc-ft-epochs=0 --amc-prune-pattern=channels --amc-prune-method=fm-reconstruction --amc-agent-algo=DDPG --amc-cfg=auto_compression_channels.yaml --amc-rllib=hanlab -j=1
"""


import os
import logging
import traceback
from functools import partial
import distiller
from environment import DistillerWrapperEnvironment, Observation
import distiller.apputils as apputils
import distiller.apputils.image_classifier as classifier
from rewards import reward_factory


msglogger = logging.getLogger()


class AutoCompressionSampleApp(classifier.ClassifierCompressor):
    def __init__(self, args, script_dir):
        super().__init__(args, script_dir)

    def train_auto_compressor(self):
        using_fm_reconstruction = self.args.amc_prune_method == 'fm-reconstruction'
        fixed_subset, sequential = (using_fm_reconstruction, using_fm_reconstruction)
        msglogger.info("AMC: fixed_subset=%s\tsequential=%s" % (fixed_subset, sequential))
        train_loader, val_loader, test_loader = classifier.load_data(self.args, fixed_subset, sequential)

        self.args.display_confusion = False
        validate_fn = partial(classifier.test, test_loader=val_loader, criterion=self.criterion,
                              loggers=self.pylogger, args=self.args, activations_collectors=None)
        train_fn = partial(classifier.train, train_loader=train_loader, criterion=self.criterion,
                           loggers=self.pylogger, args=self.args)

        save_checkpoint_fn = partial(apputils.save_checkpoint, arch=self.args.arch, dir=msglogger.logdir)
        optimizer_data = {'lr': self.args.lr, 'momentum': self.args.momentum, 'weight_decay': self.args.weight_decay}
        return train_auto_compressor(self.model, self.args, optimizer_data, validate_fn, save_checkpoint_fn, train_fn)


def main():
    import amc_args
    # Parse arguments
    args = classifier.init_classifier_compression_arg_parser()
    args = amc_args.add_automl_args(args).parse_args()
    app = AutoCompressionSampleApp(args, script_dir=os.path.dirname(__file__))
    return app.train_auto_compressor()

    
def train_auto_compressor(model, args, optimizer_data, validate_fn, save_checkpoint_fn, train_fn):
    dataset = args.dataset
    arch = args.arch
    num_ft_epochs = args.amc_ft_epochs
    action_range = args.amc_action_range

    config_verbose(False)

    # Read the experiment configuration
    amc_cfg_fname = args.amc_cfg_file
    if not amc_cfg_fname:
        raise ValueError("You must specify a valid configuration file path using --amc-cfg")

    with open(amc_cfg_fname, 'r') as cfg_file:
        compression_cfg = distiller.utils.yaml_ordered_load(cfg_file)

    if not args.amc_rllib:
        raise ValueError("You must set --amc-rllib to a valid value")

    #rl_lib = compression_cfg["rl_lib"]["name"]
    #msglogger.info("Executing AMC: RL agent - %s   RL library - %s", args.amc_agent_algo, rl_lib)

    # Create a dictionary of parameters that Coach will handover to DistillerWrapperEnvironment
    # Once it creates it.
    services = distiller.utils.MutableNamedTuple({
            'validate_fn': validate_fn,
            'save_checkpoint_fn': save_checkpoint_fn,
            'train_fn': train_fn})

    app_args = distiller.utils.MutableNamedTuple({
            'dataset': dataset,
            'arch': arch,
            'optimizer_data': optimizer_data,
            'seed': args.seed})

    ddpg_cfg = distiller.utils.MutableNamedTuple({
            'heatup_noise': 0.5,
            'initial_training_noise': 0.5,
            'training_noise_decay': 0.95,
            'num_heatup_episodes': args.amc_heatup_episodes,
            'num_training_episodes': args.amc_training_episodes,
            'actor_lr': 1e-4,
            'critic_lr': 1e-3})

    amc_cfg = distiller.utils.MutableNamedTuple({
            'modules_dict': compression_cfg["network"],  # dict of modules, indexed by arch name
            'save_chkpts': args.amc_save_chkpts,
            'protocol': args.amc_protocol,
            'agent_algo': args.amc_agent_algo,
            'num_ft_epochs': num_ft_epochs,
            'action_range': action_range,
            'reward_frequency': args.amc_reward_frequency,
            'ft_frequency': args.amc_ft_frequency,
            'pruning_pattern':  args.amc_prune_pattern,
            'pruning_method': args.amc_prune_method,
            'group_size': args.amc_group_size,
            'n_points_per_fm': args.amc_fm_reconstruction_n_pts,
            'ddpg_cfg': ddpg_cfg,
            'ranking_noise': args.amc_ranking_noise})

    #net_wrapper = NetworkWrapper(model, app_args, services)
    #return sample_networks(net_wrapper, services)

    amc_cfg.target_density = args.amc_target_density
    amc_cfg.reward_fn, amc_cfg.action_constrain_fn = reward_factory(args.amc_protocol)

    def create_environment():
        env = DistillerWrapperEnvironment(model, app_args, amc_cfg, services)
        env.amc_cfg.ddpg_cfg.replay_buffer_size = amc_cfg.ddpg_cfg.num_heatup_episodes * env.steps_per_episode
        return env

    env1 = create_environment()

    if args.amc_rllib == "spinningup":
        from rl_libs.spinningup import spinningup_if
        rl = spinningup_if.RlLibInterface()
        env2 = create_environment()
        steps_per_episode = env1.steps_per_episode
        rl.solve(env1, env2)
    elif args.amc_rllib == "hanlab":
        from rl_libs.hanlab import hanlab_if
        rl = hanlab_if.RlLibInterface()
        args.observation_len = len(Observation._fields)
        rl.solve(env1, args)
    elif args.amc_rllib == "coach":
        from rl_libs.coach import coach_if
        rl = coach_if.RlLibInterface()
        env_cfg = {'model': env1.model,
                   'app_args': env1.app_args,
                   'amc_cfg': env1.amc_cfg,
                   'services': env1.services}
        steps_per_episode = env1.steps_per_episode
        rl.solve(**env_cfg, steps_per_episode=steps_per_episode)
    elif args.amc_rllib == "random":
        from rl_libs.random import random_if
        rl = random_if.RlLibInterface()
        return rl.solve(env1)
    else:
        raise ValueError("unsupported rl library: ", args.amc_rllib)


def config_verbose(verbose, display_summaries=False):
    if verbose:
        loglevel = logging.DEBUG
    else:
        loglevel = logging.INFO
        logging.getLogger().setLevel(logging.WARNING)
    for module in ["examples.auto_compression.amc",
                   "distiller.apputils.image_classifier",
                   "distiller.thinning",
                   "distiller.pruning.ranked_structures_pruner"]:
        logging.getLogger(module).setLevel(loglevel)

    # display training progress summaries
    summaries_lvl = logging.INFO if display_summaries else logging.WARNING
    logging.getLogger("examples.auto_compression.amc.summaries").setLevel(summaries_lvl)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n-- KeyboardInterrupt --")
    except Exception as e:
        if msglogger is not None:
            # We catch unhandled exceptions here in order to log them to the log file
            # However, using the msglogger as-is to do that means we get the trace twice in stdout - once from the
            # logging operation and once from re-raising the exception. So we remove the stdout logging handler
            # before logging the exception
            handlers_bak = msglogger.handlers
            msglogger.handlers = [h for h in msglogger.handlers if type(h) != logging.StreamHandler]
            msglogger.error(traceback.format_exc())
            msglogger.handlers = handlers_bak
        raise
    finally:
        if msglogger is not None and hasattr(msglogger, 'log_filename'):
            msglogger.info('')
            msglogger.info('Log file for this run: ' + os.path.realpath(msglogger.log_filename))