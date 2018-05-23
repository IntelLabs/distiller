from agents.ddpg_agent import DDPGAgentParameters
from graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from graph_managers.graph_manager import ScheduleParameters
from base_parameters import VisualizationParameters
from core_types import TrainingSteps, EnvironmentEpisodes, EnvironmentSteps
from environments.gym_environment import MujocoInputFilter, GymEnvironmentParameters, MujocoOutputFilter


####################
# Block Scheduling #
####################
schedule_params = ScheduleParameters()
schedule_params.improve_steps = TrainingSteps(10000000000)
schedule_params.steps_between_evaluation_periods = EnvironmentEpisodes(20)
schedule_params.evaluation_steps = EnvironmentEpisodes(1)
schedule_params.heatup_steps = EnvironmentSteps(1000)

#####################
# DDPG Agent Params #
#####################
agent_params = DDPGAgentParameters()
agent_params.algorithm.num_consecutive_playing_steps = EnvironmentSteps(1)
agent_params.input_filter = MujocoInputFilter()
agent_params.output_filter = MujocoOutputFilter()

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
