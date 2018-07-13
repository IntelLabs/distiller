from agents.ddpg_agent import DDPGAgentParameters
from graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from graph_managers.graph_manager import ScheduleParameters
from base_parameters import VisualizationParameters
from core_types import EnvironmentEpisodes, EnvironmentSteps
from environments.gym_environment import MujocoInputFilter, GymEnvironmentParameters, MujocoOutputFilter
from exploration_policies.additive_noise import AdditiveNoiseParameters
from exploration_policies.truncated_normal import TruncatedNormalParameters
from schedules import ConstantSchedule, PieceWiseSchedule, ExponentialSchedule
from memories.memory import MemoryGranularity
from architectures.tensorflow_components.architecture import Dense

####################
# Block Scheduling #
####################
schedule_params = ScheduleParameters()
schedule_params.improve_steps = EnvironmentEpisodes(400)
if True:
    schedule_params.steps_between_evaluation_periods = EnvironmentEpisodes(10)
    schedule_params.evaluation_steps = EnvironmentEpisodes(3)
else:
    schedule_params.steps_between_evaluation_periods = EnvironmentEpisodes(1)
    schedule_params.evaluation_steps = EnvironmentEpisodes(1)
schedule_params.heatup_steps = EnvironmentSteps(2)

#####################
# DDPG Agent Params #
#####################
agent_params = DDPGAgentParameters()
agent_params.network_wrappers['actor'].input_embedders_parameters['observation'].scheme = [Dense([300])]
agent_params.network_wrappers['actor'].middleware_parameters.scheme = [Dense([300])]
agent_params.network_wrappers['critic'].input_embedders_parameters['observation'].scheme = [Dense([300])]
agent_params.network_wrappers['critic'].middleware_parameters.scheme = [Dense([300])]
agent_params.network_wrappers['critic'].input_embedders_parameters['action'].scheme = [Dense([300])]
#agent_params.network_wrappers['critic'].clip_gradients = 100
#agent_params.network_wrappers['actor'].clip_gradients = 100

agent_params.algorithm.rate_for_copying_weights_to_target = 0.01  # Tau pg. 11
agent_params.memory.max_size = (MemoryGranularity.Transitions, 2000)
# agent_params.memory.max_size = (MemoryGranularity.Episodes, 2000)
agent_params.exploration = TruncatedNormalParameters() # AdditiveNoiseParameters()
steps_per_episode = 13
agent_params.exploration.noise_percentage_schedule = PieceWiseSchedule([(ConstantSchedule(0.5), EnvironmentSteps(100*steps_per_episode)),
                                                                        (ExponentialSchedule(0.5, 0, 0.95), EnvironmentSteps(350*steps_per_episode))])
agent_params.algorithm.num_consecutive_playing_steps = EnvironmentSteps(1)
agent_params.input_filter = MujocoInputFilter()
agent_params.output_filter = MujocoOutputFilter()
# agent_params.network_wrappers['actor'].learning_rate = 0.0001
# agent_params.network_wrappers['critic'].learning_rate = 0.0001
# These seem like good values for Reward = -Error
agent_params.network_wrappers['actor'].learning_rate = 0.0001
agent_params.network_wrappers['critic'].learning_rate = 0.0001
# agent_params.network_wrappers['actor'].learning_rate = 0.1
# agent_params.network_wrappers['critic'].learning_rate = 0.1
# agent_params.network_wrappers['actor'].learning_rate =  0.000001
# agent_params.network_wrappers['critic'].learning_rate = 0.000001

##############################
#      Gym                   #
##############################
env_params = GymEnvironmentParameters()
#env_params.level = '/home/cvds_lab/nzmora/pytorch_workspace/distiller/examples/automated_deep_compression/gym_env/distiller_adc/distiller_adc.py:AutomatedDeepCompression'
# This path works when training from Coach
#env_params.level = '../distiller/examples/automated_deep_compression/gym_env/distiller_adc/distiller_adc.py:AutomatedDeepCompression'
# This path works when training from Distiller
#env_params.level = '../automated_deep_compression/gym_env/distiller_adc/distiller_adc.py:AutomatedDeepCompression'
env_params.level = '../automated_deep_compression/ADC.py:CNNEnvironment'


vis_params = VisualizationParameters()

graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=vis_params)
