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
from rl_coach.core_types import EnvironmentSteps
from rl_coach.schedules import ConstantSchedule, PieceWiseSchedule, ExponentialSchedule
import logging
import os

msglogger = logging.getLogger()


class RlLibInterface(object):
    """Interface to the Coach RL algorithm framework"""

    def solve(self, model, app_args, amc_cfg, services, args, steps_per_episode):
        msglogger.info("AMC: Using coach")

        amc_cfg.heatup_noise = 0.5
        amc_cfg.initial_training_noise = 0.5
        amc_cfg.training_noise_decay = 0.996  # 0.998
        amc_cfg.num_heatup_epochs = args.amc_heatup_epochs
        amc_cfg.num_training_epochs = args.amc_training_epochs
        training_noise_duration = amc_cfg.num_training_epochs * steps_per_episode
        heatup_duration = amc_cfg.num_heatup_epochs * steps_per_episode


        # When we import the graph_manager from the ADC_DDPG preset, we implicitly instruct
        # Coach to create and use our DistillerWrapperEnvironment environment.
        # So Distiller calls Coach, which creates the environment, trains the agent, and ends.
        if args.amc_agent_algo == "DDPG":
            from examples.automated_deep_compression.presets.ADC_DDPG import graph_manager, agent_params
            agent_params.exploration.noise_percentage_schedule = PieceWiseSchedule([
                (ConstantSchedule(amc_cfg.heatup_noise), EnvironmentSteps(heatup_duration)),
                (ExponentialSchedule(amc_cfg.initial_training_noise, 0, amc_cfg.training_noise_decay),
                 EnvironmentSteps(training_noise_duration))])
            # agent_params.exploration.noise_percentage_schedule = ConstantSchedule(0)
        elif "ClippedPPO" in app_args.amc_agent_algo:
            from examples.automated_deep_compression.presets.ADC_ClippedPPO import graph_manager, agent_params

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