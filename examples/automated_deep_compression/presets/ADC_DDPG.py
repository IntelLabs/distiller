from rl_coach.agents.ddpg_agent import DDPGAgentParameters
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.graph_managers.graph_manager import ScheduleParameters
from rl_coach.base_parameters import VisualizationParameters
from rl_coach.core_types import EnvironmentEpisodes, EnvironmentSteps
from rl_coach.environments.gym_environment import MujocoInputFilter, GymEnvironmentParameters, MujocoOutputFilter
from rl_coach.exploration_policies.additive_noise import AdditiveNoiseParameters
from rl_coach.exploration_policies.truncated_normal import TruncatedNormalParameters
from rl_coach.schedules import ConstantSchedule, PieceWiseSchedule, ExponentialSchedule
from rl_coach.memories.memory import MemoryGranularity
from rl_coach.base_parameters import EmbedderScheme
from rl_coach.architectures.tensorflow_components.architecture import Dense


####################
# Block Scheduling #
####################
schedule_params = ScheduleParameters()
schedule_params.improve_steps = EnvironmentEpisodes(400)
schedule_params.steps_between_evaluation_periods = EnvironmentEpisodes(1000)
schedule_params.evaluation_steps = EnvironmentEpisodes(0)
schedule_params.heatup_steps = EnvironmentSteps(2)

#####################
# DDPG Agent Params #
#####################
agent_params = DDPGAgentParameters()
agent_params.network_wrappers['actor'].input_embedders_parameters['observation'].scheme = [Dense([300])]
agent_params.network_wrappers['actor'].middleware_parameters.scheme = [Dense([300])]
agent_params.network_wrappers['critic'].input_embedders_parameters['observation'].scheme = [Dense([300])]
agent_params.network_wrappers['critic'].middleware_parameters.scheme = [Dense([300])]
agent_params.network_wrappers['critic'].input_embedders_parameters['action'].scheme = EmbedderScheme.Empty
agent_params.network_wrappers['actor'].heads_parameters[0].activation_function = 'sigmoid'
#agent_params.network_wrappers['critic'].clip_gradients = 100
#agent_params.network_wrappers['actor'].clip_gradients = 100

agent_params.algorithm.rate_for_copying_weights_to_target = 0.01  # Tau pg. 11
agent_params.algorithm.num_steps_between_copying_online_weights_to_target = EnvironmentSteps(1)
agent_params.algorithm.discount = 1
agent_params.memory.max_size = (MemoryGranularity.Transitions, 2000)
agent_params.exploration =  TruncatedNormalParameters() # AdditiveNoiseParameters()
steps_per_episode = 13
agent_params.exploration.noise_percentage_schedule = PieceWiseSchedule([(ConstantSchedule(0.5), EnvironmentSteps(100*steps_per_episode)),
                                                                        (ExponentialSchedule(0.5, 0, 0.996), EnvironmentSteps(300*steps_per_episode))])
agent_params.algorithm.num_consecutive_playing_steps = EnvironmentSteps(1)
agent_params.input_filter = MujocoInputFilter()
agent_params.output_filter = MujocoOutputFilter()
agent_params.network_wrappers['actor'].learning_rate = 0.0001
agent_params.network_wrappers['critic'].learning_rate = 0.001

##############################
#      Gym                   #
##############################
env_params = GymEnvironmentParameters()
env_params.level = '../automated_deep_compression/ADC.py:CNNEnvironment'


vis_params = VisualizationParameters()
vis_params.dump_parameters_documentation = False

graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=vis_params)
