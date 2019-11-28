from rl_coach.agents.td3_agent import TD3AgentParameters
from rl_coach.architectures.layers import Dense
from rl_coach.base_parameters import VisualizationParameters, PresetValidationParameters, EmbedderScheme
from rl_coach.core_types import EnvironmentEpisodes, EnvironmentSteps
from rl_coach.environments.environment import SingleLevelSelection
from rl_coach.environments.gym_environment import GymVectorEnvironment, mujoco_v2
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.graph_managers.graph_manager import ScheduleParameters
from rl_coach.exploration_policies.truncated_normal import TruncatedNormalParameters
from rl_coach.exploration_policies.additive_noise import AdditiveNoiseParameters

####################
# Graph Scheduling #
####################
schedule_params = ScheduleParameters()
schedule_params.improve_steps = EnvironmentEpisodes(800)
schedule_params.steps_between_evaluation_periods = EnvironmentEpisodes(5000)
schedule_params.evaluation_steps = EnvironmentEpisodes(0)  # Neta: 0
schedule_params.heatup_steps = EnvironmentEpisodes(100)

#########
# Agent #
#########
agent_params = TD3AgentParameters()
agent_params.network_wrappers['actor'].input_embedders_parameters['observation'].scheme = [Dense(400)]
agent_params.network_wrappers['actor'].middleware_parameters.scheme = [Dense(300)]

agent_params.network_wrappers['critic'].input_embedders_parameters['observation'].scheme = EmbedderScheme.Empty
agent_params.network_wrappers['critic'].input_embedders_parameters['action'].scheme = EmbedderScheme.Empty
agent_params.network_wrappers['critic'].middleware_parameters.scheme = [Dense(400), Dense(300)]

agent_params.exploration = TruncatedNormalParameters()
agent_params.exploration.noise_as_percentage_from_action_space = False
agent_params.exploration.evaluation_noise = 0  # Neta new
agent_params.algorithm.use_target_network_for_evaluation = True
agent_params.algorithm.act_for_full_episodes = True



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