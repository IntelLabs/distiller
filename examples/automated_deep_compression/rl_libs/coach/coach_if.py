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

from rl_coach.base_parameters import TaskParameters
from rl_coach.core_types import EnvironmentSteps, EnvironmentEpisodes, TrainingSteps
from rl_coach.schedules import ConstantSchedule, PieceWiseSchedule, ExponentialSchedule
from rl_coach.memories.memory import MemoryGranularity
import logging
import os

msglogger = logging.getLogger()


class RlLibInterface(object):
    """Interface to the Coach RL algorithm framework"""

    def solve(self, model, app_args, amc_cfg, services, args, steps_per_episode):
        msglogger.info("AMC: Using coach")

        # When we import the graph_manager from the ADC_DDPG preset, we implicitly instruct
        # Coach to create and use our DistillerWrapperEnvironment environment.
        # So Distiller calls Coach, which creates the environment, trains the agent, and ends.
        if args.amc_agent_algo == "DDPG":
            from examples.automated_deep_compression.rl_libs.coach.presets.ADC_DDPG import (graph_manager, 
                                                                                            agent_params)
            graph_manager.agent_params.exploration.noise_schedule = ExponentialSchedule(amc_cfg.ddpg_cfg.initial_training_noise, 
                                                                          0, 
                                                                          amc_cfg.ddpg_cfg.training_noise_decay)
            # Number of iterations to train 
            graph_manager.agent_params.algorithm.num_consecutive_training_steps = steps_per_episode
            graph_manager.agent_params.algorithm.num_steps_between_copying_online_weights_to_target = TrainingSteps(1)
            # Heatup
            graph_manager.heatup_steps = EnvironmentEpisodes(amc_cfg.ddpg_cfg.num_heatup_epochs)
            amc_cfg.ddpg_cfg.num_heatup_epochs
            # Replay buffer size
            graph_manager.agent_params.memory.max_size = (MemoryGranularity.Transitions, amc_cfg.ddpg_cfg.replay_buffer_size)
        elif "ClippedPPO" in args.amc_agent_algo:
            from examples.automated_deep_compression.rl_libs.coach.presets.ADC_ClippedPPO import graph_manager, agent_params
        elif "TD3" in args.amc_agent_algo:
            from examples.automated_deep_compression.rl_libs.coach.presets.ADC_TD3 import graph_manager, agent_params

        # Number of training steps
        graph_manager.improve_steps = EnvironmentEpisodes(amc_cfg.ddpg_cfg.num_training_epochs)
        graph_manager.steps_between_evaluation_periods = EnvironmentEpisodes(amc_cfg.ddpg_cfg.num_training_epochs)
        # These parameters are passed to the Distiller environment
        env_cfg  = {'model': model, 
                    'app_args': app_args,
                    'amc_cfg': amc_cfg,
                    'services': services}
        graph_manager.env_params.additional_simulator_parameters = env_cfg

        coach_logs_dir = os.path.join(msglogger.logdir, 'coach')
        os.mkdir(coach_logs_dir)
        task_parameters = TaskParameters(experiment_path=coach_logs_dir)
        graph_manager.create_graph(task_parameters)
        graph_manager.improve()