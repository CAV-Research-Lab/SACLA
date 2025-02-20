from typing import NamedTuple
import flax
import numpy as np
from flax.training.train_state import TrainState
import gymnasium as gym
import numpy as np
import orbax.checkpoint as ocp

from prob_lyap.utils.wrappers_rd import NoneWrapper, UnseenRandomDelayWrapper, AugmentedRandomDelayWrapper
from prob_lyap.objectives import *
import enum
from dataclasses import dataclass

class RLTrainState(TrainState):  # type: ignore[misc]
    target_params: flax.core.FrozenDict  # type: ignore[misc]

class ReplayBufferSamplesNp(NamedTuple):
    observations: np.ndarray
    actions: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray
    rewards: np.ndarray
    achieved_goals: np.ndarray
    desired_goals: np.ndarray
    next_achieved_goals: np.ndarray
    next_desired_goals: np.ndarray

@dataclass
class LyapConf:
    seed: int = np.random.randint(100)
    n_hidden: int = 32
    n_layers: int = 1
    lyap_lr: float = 0.0005
    wm_lr: float = 0.0005
    actor_lr: float = 0.0005
    ckpt_every: int = 20_000
    total_timesteps: int = 100_000
    env_id: str = "FetchReach-v2"
    delay_type: gym.Wrapper = NoneWrapper
    act_delay: range = range(0,1)
    obs_delay: range = range(0,1)
    ckpt_dir: str = "default"
    objective: str = "adverserial"
    beta: float = 0.5
    debug: bool = False
    logging: str="none"

    def __str__(self) -> str:
        return "\n".join([f"{k}: {v}" for k,v in self.__dict__.items()])

    # def __dict__(self) -> dict:
    #     return self.__dict__.items()

class CustomTrainState(TrainState):
    learning_rate: float

class MyStateHandler(ocp.pytree_checkpoint_handler.TypeHandler):
    pass

'''
    lyap_config = { # Make a class
    "seed": seed,
    "n_hidden": n_hidden,
    "learning_rate": learning_rate,
    "ckpt_every": ckpt_every,
    "total_timesteps": total_timesteps,
    "env_id": env_id,
    "delay_type": DELAY_CLASSES[delay_type], # Class i.e. AugmentedRandomDelayWrapper etc OR NoneWrapper
    "delay_config": {"obs_delay_range": range(d_,1), "act_delay_range": range(0,1)},
    "ckpt_dir": "default" # TODO: Only supports default for now
    }

'''
