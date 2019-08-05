from rl_coach.agents.ddpg_agent import DDPGAgentParameters
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.graph_managers.graph_manager import ScheduleParameters
from rl_coach.base_parameters import VisualizationParameters
from rl_coach.core_types import EnvironmentEpisodes, EnvironmentSteps
from rl_coach.environments.gym_environment import GymVectorEnvironment
from rl_coach.exploration_policies.truncated_normal import TruncatedNormalParameters
from rl_coach.exploration_policies.additive_noise import AdditiveNoiseParameters
from rl_coach.base_parameters import EmbedderScheme
from rl_coach.architectures.tensorflow_components.layers import Dense
from rl_coach.base_parameters import EmbeddingMergerType
from rl_coach.filters.filter import InputFilter
# !!!! Enable when using branch "distiller-AMC-induced-changes" 
from rl_coach.filters.reward import RewardEwmaNormalizationFilter
import numpy as np

####################
# Graph Scheduling #
####################
schedule_params = ScheduleParameters()
schedule_params.steps_between_evaluation_periods = EnvironmentEpisodes(0)
schedule_params.evaluation_steps = EnvironmentEpisodes(0)

#####################
# DDPG Agent Params #
#####################
agent_params = DDPGAgentParameters()
agent_params.network_wrappers['actor'].input_embedders_parameters['observation'].scheme = [Dense(300)]
agent_params.network_wrappers['actor'].middleware_parameters.scheme = [Dense(300)]
agent_params.network_wrappers['actor'].heads_parameters[0].activation_function = 'sigmoid'
agent_params.network_wrappers['critic'].input_embedders_parameters['observation'].scheme = [Dense(300)]
agent_params.network_wrappers['critic'].middleware_parameters.scheme = [Dense(300)]
agent_params.network_wrappers['critic'].input_embedders_parameters['action'].scheme = [Dense(300)]

agent_params.network_wrappers['critic'].embedding_merger_type = EmbeddingMergerType.Sum

agent_params.network_wrappers['actor'].optimizer_type = 'Adam'
agent_params.network_wrappers['actor'].adam_optimizer_beta1 = 0.9
agent_params.network_wrappers['actor'].adam_optimizer_beta2 = 0.999
agent_params.network_wrappers['actor'].optimizer_epsilon = 1e-8
#agent_params.network_wrappers['actor'].wd = 0

agent_params.network_wrappers['critic'].optimizer_type = 'Adam'
agent_params.network_wrappers['critic'].adam_optimizer_beta1 = 0.9
agent_params.network_wrappers['critic'].adam_optimizer_beta2 = 0.999
agent_params.network_wrappers['critic'].optimizer_epsilon = 1e-8

agent_params.network_wrappers['actor'].learning_rate = 1e-4
agent_params.network_wrappers['critic'].learning_rate = 1e-3

# !!!! Enable when using branch "distiller-AMC-induced-changes"
agent_params.algorithm.override_episode_rewards_with_the_last_transition_reward = True

agent_params.algorithm.rate_for_copying_weights_to_target = 0.01  # Tau pg. 11
#agent_params.algorithm.num_steps_between_copying_online_weights_to_target = EnvironmentSteps(1)
agent_params.algorithm.heatup_using_network_decisions = False # We want uniform-random samples during heatup
agent_params.algorithm.discount = 1
agent_params.algorithm.use_non_zero_discount_for_terminal_states = True

agent_params.exploration = TruncatedNormalParameters()
#agent_params.exploration = AdditiveNoiseParameters()
agent_params.exploration.noise_as_percentage_from_action_space = False
agent_params.exploration.evaluation_noise = 0  # Neta new
agent_params.algorithm.use_target_network_for_evaluation = True
agent_params.algorithm.act_for_full_episodes = True

# !!!! Enable when using branch "distiller-AMC-induced-changes"
agent_params.algorithm.update_pre_network_filters_state_on_train = True
agent_params.algorithm.update_pre_network_filters_state_on_inference = False

# !!!! Enable when using branch "distiller-AMC-induced-changes" 
agent_params.pre_network_filter = InputFilter()
agent_params.pre_network_filter.add_reward_filter('ewma_norm', RewardEwmaNormalizationFilter(alpha=0.5))

##############################
#      Gym                   #
##############################
env_params = GymVectorEnvironment()
env_params.level = './environment.py:DistillerWrapperEnvironment'

vis_params = VisualizationParameters()
vis_params.dump_parameters_documentation = False
vis_params.render = True
vis_params.native_rendering = True
vis_params.dump_signals_to_csv_every_x_episodes = 1
graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=vis_params)
