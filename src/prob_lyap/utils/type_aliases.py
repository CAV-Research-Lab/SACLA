from typing import NamedTuple
import flax
import numpy as np
from flax.training.train_state import TrainState
import gymnasium as gym
import numpy as np
import orbax.checkpoint as ocp

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
    n_hidden: int = 16
    n_layers: int = 1
    lyap_lr: float = 0.0005
    wm_lr: float = 0.0005
    actor_lr: float = 0.0005
    ckpt_every: int = 20_000
    total_timesteps: int = 100_000
    env_id: str = "FetchReach-v2"
    ckpt_dir: str = "default"
    beta: float = 0.5
    debug: bool = False
    logging: str="none"

    def __str__(self) -> str:
        return "\n".join([f"{k}: {v}" for k,v in self.__dict__.items()])

class CustomTrainState(TrainState):
    learning_rate: float

class MyStateHandler(ocp.PyTreeCheckpointHandler):
    pass


