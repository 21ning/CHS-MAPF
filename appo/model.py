import json
import os
import torch
from copy import deepcopy
from sample_factory.algorithms.appo.learner import LearnerWorker
from sample_factory.algorithms.appo.model import create_actor_critic
from sample_factory.utils.utils import AttrDict
from sample_factory.algorithms.appo.actor_worker import transform_dict_observations
from sample_factory.algorithms.appo.model_utils import get_hidden_size
from pydantic import BaseModel
from pydantic import Extra
from appo.utils.epom_config import Environment, Experiment, ExperimentSettings
from appo.utils.encoder import ResnetEncoder
from argparse import Namespace
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
from sample_factory.utils.timing import Timing

from sample_factory.envs.create_env import create_env
from sample_factory.algorithms.utils.algo_utils import EXTRA_EPISODIC_STATS_PROCESSING, EXTRA_PER_POLICY_SUMMARIES
from sample_factory.envs.env_registry import global_env_registry
from pogema import pogema_v0
from appo.utils.wrappers import  MatrixObservationWrapper
from appo.utils.grid_memory import GridMemoryWrapper,MultipleGridMemory
def make_env(env_cfg: Environment = Environment()):
    env = make_pomapf(grid_config=env_cfg.grid_config)
    return env

def create_pogema_env(full_env_name, cfg=None, env_config=None):
    environment_config: Environment = Environment(**cfg.full_config['environment'])
    env = make_env(environment_config)
    gm_radius = environment_config.grid_memory_obs_radius
    env = GridMemoryWrapper(env, obs_radius=gm_radius if gm_radius else environment_config.grid_config.obs_radius)
    env = MatrixObservationWrapper(env)
    return env


def make_pomapf(grid_config, with_animations=False):
    grid_config.auto_reset = False
    env = pogema_v0(grid_config)
    return env

def register_custom_components():
    global_env_registry().register_env(
        env_name_prefix='POMAPF',
        make_env_func=create_pogema_env,
    )

    EXTRA_EPISODIC_STATS_PROCESSING.append(pogema_extra_episodic_stats_processing)
    EXTRA_PER_POLICY_SUMMARIES.append(pogema_extra_summaries)


def pogema_extra_episodic_stats_processing(policy_id, stat_key, stat_value, cfg):
    pass


def pogema_extra_summaries(policy_id, policy_avg_stats, env_steps, summary_writer, cfg):
    for key in policy_avg_stats:
        if key in ['reward', 'len', 'true_reward', 'Done']:
            continue

        avg = np.mean(np.array(policy_avg_stats[key][policy_id]))
        summary_writer.add_scalar(key, avg, env_steps)
        log.debug(f'{policy_id}-{key}: {round(float(avg), 3)}')


register_custom_components()

def validate_config(config):
    exp = Experiment(**config)
    flat_config = Namespace(**exp.async_ppo.dict(),
                            **exp.experiment_settings.dict(),
                            **exp.global_settings.dict(),
                            **exp.evaluation.dict(),
                            full_config=exp.dict()
                            )
                              
    return exp, flat_config

class AlgoBase(BaseModel):
    name: str = None
    num_process: int = 3
    device: str = 'cuda'


class EpomConfig(AlgoBase, extra=Extra.forbid):
    path_to_weights: str = "weights/epom/checkpoint_p0"
    name: Literal['EPOM'] = 'EPOM'

class EPOM:
    def __init__(self, algo_cfg):
        self.algo_cfg: EpomConfig = algo_cfg
        self.path = algo_cfg.path_to_weights
        # Set device
        self.device = torch.device(algo_cfg.device if torch.cuda.is_available() else 'cpu')

        # Load configuration
        config_path = os.path.join(self.path, 'cfg.json')
        with open(config_path, "r") as f:
            config = json.load(f)
        exp, flat_config = validate_config(config['full_config'])
        algo_cfg = flat_config

        # Create actor-critic model
        # Assuming observation_space and action_space are known or can be loaded
        env = create_env(algo_cfg.env, cfg=algo_cfg, env_config={})
        observation_space = env.observation_space
        action_space = env.action_space
        actor_critic = create_actor_critic(algo_cfg, observation_space, action_space)
        exp_set = ExperimentSettings()
        cfg = {'full_config': {'experiment_settings': exp_set.dict()}, **exp_set.dict()}
        cfg = Namespace(**cfg)
        #ResnetEncoder(cfg, observation_space, Timing())
        actor_critic.model_to_device(self.device)
        # Load model checkpoint
        checkpoints_path = os.path.join(self.path)
        checkpoints_path =  "weights/epom/checkpoint_p0"
        checkpoints = LearnerWorker.get_checkpoints(checkpoints_path)
        checkpoint_dict = LearnerWorker.load_checkpoint(checkpoints, self.device)

        actor_critic.load_state_dict(checkpoint_dict['model'])
        self.ppo = actor_critic
        self.rnn_states = None
        self.cfg = algo_cfg
        self.mgm = MultipleGridMemory()
        self._step = 0
        
    def act(self, observations, rewards=None, dones=None, infos=None, explored_map=None, agents_positions= None):
        observations = deepcopy(observations)
        #print(observations)
        if self.rnn_states is None or len(self.rnn_states) != len(observations):
            self.rnn_states = torch.zeros([len(observations), get_hidden_size(self.cfg)], dtype=torch.float32,
                                          device=self.device)
        env_cfg: Environment = Environment(**self.cfg.full_config['environment'])
        self.mgm.update(observations, agents_positions, explored_map)
        gm_radius = env_cfg.grid_memory_obs_radius
        #print(agents_positions)
        self.mgm.modify_observation(observations, obs_radius=gm_radius if gm_radius else env_cfg.grid_config.obs_radius, explored_map = explored_map)
        observations = MatrixObservationWrapper.to_matrix(observations)
        #print(observations[0])
        with torch.no_grad():
            #进行一个预处理
            obs_torch = AttrDict(transform_dict_observations(observations))
            for key, x in obs_torch.items():
                obs_torch[key] = torch.from_numpy(x).to(self.device).float()
            policy_outputs = self.ppo(obs_torch, self.rnn_states, with_action_distribution=True)
            self.rnn_states = policy_outputs.rnn_states
            actions = policy_outputs.actions

        self._step += 1
        result = actions.cpu().numpy()
        return result

