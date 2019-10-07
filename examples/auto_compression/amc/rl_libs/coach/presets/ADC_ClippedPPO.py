from rl_coach.agents.clipped_ppo_agent import ClippedPPOAgentParameters
from rl_coach.architectures.layers import Dense
from rl_coach.base_parameters import VisualizationParameters, DistributedCoachSynchronizationType
from rl_coach.core_types import TrainingSteps, EnvironmentEpisodes, EnvironmentSteps
from rl_coach.environments.gym_environment import GymVectorEnvironment
from rl_coach.architectures.head_parameters import PPOHeadParameters, VHeadParameters
from rl_coach.filters.filter import InputFilter
from rl_coach.filters.observation.observation_normalization_filter import ObservationNormalizationFilter
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.graph_managers.graph_manager import ScheduleParameters
from rl_coach.schedules import LinearSchedule

####################
# Graph Scheduling #
####################

schedule_params = ScheduleParameters()
schedule_params.improve_steps = TrainingSteps(10000000)
schedule_params.steps_between_evaluation_periods = EnvironmentSteps(2048)
schedule_params.evaluation_steps = EnvironmentEpisodes(5)
schedule_params.heatup_steps = EnvironmentSteps(0)

#########
# Agent #
#########
agent_params = ClippedPPOAgentParameters()


agent_params.network_wrappers['main'].learning_rate = 0.0001  #  0.0003
agent_params.network_wrappers['main'].input_embedders_parameters['observation'].activation_function = 'tanh'
agent_params.network_wrappers['main'].input_embedders_parameters['observation'].scheme = [Dense(64)]
agent_params.network_wrappers['main'].middleware_parameters.scheme = [Dense(64)]
agent_params.network_wrappers['main'].middleware_parameters.activation_function = 'tanh'
agent_params.network_wrappers['main'].batch_size = 64
agent_params.network_wrappers['main'].optimizer_epsilon = 1e-5
agent_params.network_wrappers['main'].adam_optimizer_beta2 = 0.999

agent_params.algorithm.clip_likelihood_ratio_using_epsilon = 0.2
agent_params.algorithm.num_consecutive_playing_steps.num_steps = 1024
agent_params.algorithm.clipping_decay_schedule = LinearSchedule(1.0, 0, 1000000)
# The entropy coefficient is a regularizer. A policy has maximum entropy when all policies are equally
# likely and minimum when the one action probability of the policy is dominant. The entropy coefficient
# is multiplied by the maximum possible entropy and added to loss. This helps prevent premature convergence
# of one action probability dominating the policy and preventing exploration.
# Source: https://medium.com/aureliantactics/ppo-hyperparameters-and-ranges-6fc2d29bccbe
# This value sets the regularization penalty
agent_params.algorithm.beta_entropy = 0
agent_params.algorithm.gae_lambda = 0.95
agent_params.algorithm.discount = 1
# How many epochs to train the network using supervised methods
agent_params.algorithm.optimization_epochs = 10
agent_params.algorithm.estimate_state_value_using_gae = True

# Distributed Coach synchronization type.
agent_params.algorithm.distributed_coach_synchronization_type = DistributedCoachSynchronizationType.SYNC

agent_params.pre_network_filter = InputFilter()
agent_params.pre_network_filter.add_observation_filter('observation', 'normalize_observation',
                                                        ObservationNormalizationFilter(name='normalize_observation'))

###############
# Environment #
###############
env_params = GymVectorEnvironment()
env_params.level = './environment.py:DistillerWrapperEnvironment'

vis_params = VisualizationParameters()
vis_params.dump_parameters_documentation = False
vis_params.render = True
vis_params.native_rendering = True
vis_params.dump_signals_to_csv_every_x_episodes = 1
graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=vis_params)
