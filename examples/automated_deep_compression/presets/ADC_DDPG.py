from agents.ddpg_agent import DDPGAgentParameters
from block_factories.basic_rl_factory import BasicRLFactory
from block_scheduler import BlockSchedulerParameters
from configurations import VisualizationParameters
from core_types import TrainingSteps, Episodes, EnvironmentSteps, RunPhase
from environments.gym_environment import MujocoInputFilter, Mujoco
from schedules import LinearSchedule
from environments.environment import MaxDumpMethod, SelectedPhaseOnlyDumpMethod
from environments.gym_environment import GymEnvironmentParameters

####################
# Block Scheduling #
####################
from memories.memory import MemoryGranularity

schedule_params = BlockSchedulerParameters()
schedule_params.improve_steps = TrainingSteps(10000000000)
schedule_params.steps_between_evaluation_periods = Episodes(20)
schedule_params.evaluation_steps = Episodes(1)
schedule_params.heatup_steps = EnvironmentSteps(1000)

#####################
# DDPG Agent Params #
#####################
agent_params = DDPGAgentParameters()
agent_params.algorithm.num_consecutive_playing_steps = 1
agent_params.input_filter = MujocoInputFilter()

##############################
#      Gym                   #
##############################
env_params = GymEnvironmentParameters()
env_params.level = 'distiller_adc:AutomatedDeepCompression'
env_params.render = True
#env_params.seed = 1

vis_params = VisualizationParameters()

factory = BasicRLFactory(agent_params=agent_params, env_params=env_params,
                         schedule_params=schedule_params, vis_params=vis_params)
