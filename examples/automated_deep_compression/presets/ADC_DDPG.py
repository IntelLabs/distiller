from agents.ddpg_agent import DDPGAgentParameters
from graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from graph_managers.graph_manager import ScheduleParameters
from base_parameters import VisualizationParameters
from core_types import TrainingSteps, EnvironmentEpisodes, EnvironmentSteps
from environments.gym_environment import MujocoInputFilter, GymEnvironmentParameters, MujocoOutputFilter
from exploration_policies.additive_noise import AdditiveNoiseParameters
from schedules import ConstantSchedule, PieceWiseSchedule, ExponentialSchedule

####################
# Block Scheduling #
####################
schedule_params = ScheduleParameters()
schedule_params.improve_steps = EnvironmentEpisodes(400)
schedule_params.steps_between_evaluation_periods = EnvironmentEpisodes(400)
schedule_params.evaluation_steps = EnvironmentEpisodes(0)
schedule_params.heatup_steps = EnvironmentSteps(2)

#####################
# DDPG Agent Params #
#####################
agent_params = DDPGAgentParameters()
agent_params.exploration = AdditiveNoiseParameters()
steps_per_episode = 16
agent_params.exploration.noise_percentage_schedule = PieceWiseSchedule([(ConstantSchedule(0.5), EnvironmentSteps(100*steps_per_episode)),
                                                                        (ExponentialSchedule(0.5, 0, 0.95), EnvironmentSteps(300*steps_per_episode))])
agent_params.algorithm.num_consecutive_playing_steps = EnvironmentSteps(1)
agent_params.input_filter = MujocoInputFilter()
agent_params.output_filter = MujocoOutputFilter()
agent_params.network_wrappers['actor'].learning_rate = 0.0001
agent_params.network_wrappers['critic'].learning_rate = 0.0001

##############################
#      Gym                   #
##############################
env_params = GymEnvironmentParameters()
#env_params.level = '/home/cvds_lab/nzmora/pytorch_workspace/distiller/examples/automated_deep_compression/gym_env/distiller_adc/distiller_adc.py:AutomatedDeepCompression'
# This path works when training from Coach
#env_params.level = '../distiller/examples/automated_deep_compression/gym_env/distiller_adc/distiller_adc.py:AutomatedDeepCompression'
# This path works when training from Distiller
#env_params.level = '../automated_deep_compression/gym_env/distiller_adc/distiller_adc.py:AutomatedDeepCompression'
env_params.level = '../classifier_compression/ADC.py:CNNEnvironment'


vis_params = VisualizationParameters()

graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=vis_params)
