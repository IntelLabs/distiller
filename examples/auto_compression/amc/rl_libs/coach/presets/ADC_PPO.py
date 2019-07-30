from rl_coach.agents.ppo_agent import PPOAgentParameters
from rl_coach.architectures.layers import Dense
from rl_coach.base_parameters import VisualizationParameters, PresetValidationParameters, DistributedCoachSynchronizationType
from rl_coach.core_types import TrainingSteps, EnvironmentEpisodes, EnvironmentSteps
from rl_coach.environments.environment import SingleLevelSelection
from rl_coach.environments.gym_environment import GymVectorEnvironment, mujoco_v2
from rl_coach.filters.filter import InputFilter
from rl_coach.filters.observation.observation_normalization_filter import ObservationNormalizationFilter
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.graph_managers.graph_manager import ScheduleParameters

####################
# Graph Scheduling #
####################
schedule_params = ScheduleParameters()
schedule_params.improve_steps = TrainingSteps(10000000000)
schedule_params.steps_between_evaluation_periods = EnvironmentSteps(2000)
schedule_params.evaluation_steps = EnvironmentEpisodes(1)
schedule_params.heatup_steps = EnvironmentSteps(0)

#########
# Agent #
#########
agent_params = PPOAgentParameters()
agent_params.network_wrappers['actor'].learning_rate = 0.001
agent_params.network_wrappers['critic'].learning_rate = 0.001

agent_params.network_wrappers['actor'].input_embedders_parameters['observation'].scheme = [Dense(64)]
agent_params.network_wrappers['actor'].middleware_parameters.scheme = [Dense(64)]
agent_params.network_wrappers['critic'].input_embedders_parameters['observation'].scheme = [Dense(64)]
agent_params.network_wrappers['critic'].middleware_parameters.scheme = [Dense(64)]

agent_params.input_filter = InputFilter()
agent_params.input_filter.add_observation_filter('observation', 'normalize', ObservationNormalizationFilter())

# Distributed Coach synchronization type.
agent_params.algorithm.distributed_coach_synchronization_type = DistributedCoachSynchronizationType.SYNC

###############
# Environment #
###############
env_params = GymVectorEnvironment()
env_params.level = '../automated_deep_compression/ADC.py:DistillerWrapperEnvironment'

vis_params = VisualizationParameters()
vis_params.dump_parameters_documentation = False
vis_params.render = True
vis_params.native_rendering = True
vis_params.dump_signals_to_csv_every_x_episodes = 1
graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=vis_params)


