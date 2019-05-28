from rl_coach.agents.ddpg_agent import DDPGAgentParameters
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.graph_managers.graph_manager import ScheduleParameters
from rl_coach.base_parameters import VisualizationParameters
from rl_coach.core_types import EnvironmentEpisodes, EnvironmentSteps
from rl_coach.environments.gym_environment import GymVectorEnvironment
from rl_coach.exploration_policies.truncated_normal import TruncatedNormalParameters
from rl_coach.exploration_policies.additive_noise import AdditiveNoiseParameters

from rl_coach.memories.memory import MemoryGranularity
from rl_coach.base_parameters import EmbedderScheme
from rl_coach.architectures.tensorflow_components.layers import Dense

steps_per_episode = 13

####################
# Block Scheduling #
####################
schedule_params = ScheduleParameters()
schedule_params.improve_steps = EnvironmentEpisodes(800)
schedule_params.steps_between_evaluation_periods = EnvironmentEpisodes(5)
schedule_params.evaluation_steps = EnvironmentEpisodes(1)  # Neta: 0
schedule_params.heatup_steps = EnvironmentEpisodes(100)

#####################
# DDPG Agent Params #
#####################
agent_params = DDPGAgentParameters()
agent_params.network_wrappers['actor'].input_embedders_parameters['observation'].scheme = [Dense(300)]
agent_params.network_wrappers['actor'].middleware_parameters.scheme = [Dense(300)]
agent_params.network_wrappers['actor'].heads_parameters[0].activation_function = 'sigmoid'
agent_params.network_wrappers['critic'].input_embedders_parameters['observation'].scheme = [Dense(300)]
agent_params.network_wrappers['critic'].middleware_parameters.scheme = [Dense(300)]
agent_params.network_wrappers['critic'].input_embedders_parameters['action'].scheme = EmbedderScheme.Empty

agent_params.network_wrappers['actor'].optimizer_type = 'Adam'
agent_params.network_wrappers['actor'].adam_optimizer_beta1 = 0.9
agent_params.network_wrappers['actor'].adam_optimizer_beta2 = 0.999

agent_params.network_wrappers['critic'].optimizer_type = 'Adam'
agent_params.network_wrappers['critic'].adam_optimizer_beta1 = 0.9
agent_params.network_wrappers['critic'].adam_optimizer_beta2 = 0.999


agent_params.network_wrappers['actor'].learning_rate = 0.0001  # 0.0001
agent_params.network_wrappers['critic'].learning_rate = 0.001
# agent_params.network_wrappers['critic'].clip_gradients = 100
# agent_params.network_wrappers['actor'].clip_gradients = 100

agent_params.algorithm.rate_for_copying_weights_to_target = 0.01  # Tau pg. 11
agent_params.algorithm.num_steps_between_copying_online_weights_to_target = EnvironmentSteps(1)
agent_params.algorithm.heatup_using_network_decisions = True
agent_params.algorithm.discount = 1

# Shadi
agent_params.algorithm.use_non_zero_discount_for_terminal_states = True  # <===========

# See : https://nervanasystems.github.io/coach/components/agents/policy_optimization/ddpg.html?highlight=ddpg#rl_coach.agents.ddpg_agent.DDPGAlgorithmParameters
# Replay buffer size
agent_params.memory.max_size = (MemoryGranularity.Transitions, 2000)
#agent_params.exploration = AdditiveNoiseParameters() #
agent_params.exploration = TruncatedNormalParameters()
agent_params.algorithm.use_target_network_for_evaluation = True
#agent_params.exploration.evaluation_noise_percentage = 0.01  # Neta new
agent_params.algorithm.num_consecutive_playing_steps = EnvironmentSteps(1)


##############################
#      Gym                   #
##############################
env_params = GymVectorEnvironment()
env_params.level = '../automated_deep_compression/ADC.py:DistillerWrapperEnvironment'


vis_params = VisualizationParameters()
vis_params.dump_parameters_documentation = False
vis_params.render = True
vis_params.native_rendering = True
vis_params.dump_signals_to_csv_every_x_episodes = 1
graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=vis_params)
