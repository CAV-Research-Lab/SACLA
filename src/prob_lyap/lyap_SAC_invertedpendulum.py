from functools import partial
import numpy as np
import sbx
import click
import jax
from jax._src import prng
import jax.numpy as jnp
import jax.random as random

import flax
from flax.training.train_state import TrainState
import orbax.checkpoint as ocp
from flax.training import orbax_utils
from flax.core import FrozenDict

from prob_lyap.utils.type_aliases import ReplayBufferSamplesNp, RLTrainState, CustomTrainState, LyapConf
from prob_lyap.utils import utils
from prob_lyap.lyap_func import Lyap_net
from prob_lyap.lyap_SAC import Lyap_SAC
from prob_lyap.lyap_func_InvertedPendulum import Lyap_net_IP
from prob_lyap.world_model import WorldModel
from prob_lyap.objectives import get_objective

from stable_baselines3.common.buffers import RolloutBuffer

from copy import deepcopy
from typing import Callable

from typing import Dict
import copy

class Lyap_SAC_IP(Lyap_SAC):    
    def __init__(self, lyap_config: LyapConf, *args, **kwargs):
        print("InvPend Lyap SAC")
        super().__init__(lyap_config, *args, **kwargs)
 
    @classmethod
    @partial(jax.jit, static_argnames=["cls", "gradient_steps", "objective_fn"])
    def _train(
        cls,
        gamma: float,
        tau: float,
        target_entropy: np.ndarray,
        gradient_steps: int,
        data: ReplayBufferSamplesNp,
        policy_delay_indices: flax.core.FrozenDict,
        qf_state: RLTrainState,
        actor_state: TrainState,
        ent_coef_state: TrainState,
        lyap_state: CustomTrainState, #
        wm_state: CustomTrainState, #
        key: prng.PRNGKeyArray, # typecheck
        objective_fn: Callable[[jnp.ndarray, jnp.ndarray, float], jnp.ndarray],
        beta: float,
        debug: bool
    ):
        actor_loss_value = jnp.array(0)
        for i in range(gradient_steps):

            def slice(x, step=i):
                assert x.shape[0] % gradient_steps == 0
                batch_size = x.shape[0] // gradient_steps
                return x[batch_size * step : batch_size * (step + 1)]
            
            # Update models
            lyap_state, lyap_loss, lyap_risks, v_candidates_mean, key = Lyap_net_IP.update(lyap_state,
                                                           wm_state,
                                                           actor_state,
                                                        #    slice(data.actions),
                                                        #    slice(data.achieved_goals),
                                                        #    slice(data.desired_goals),
                                                           slice(data.observations),
                                                        #    slice(data.next_observations),
                                                           key)
            
            # print(lyap_state.params)
            # assert not utils.params_equal(lyap_state.params['params'], prev_lyap_state.params['params']), "Lyap state equal after update!"
            
            # print(f"REWARDS: \n{slice(data.rewards)}\n\n")
            # print(f"lyap_risks: \n{lyap_risks}\n\n")
            # print(f"beta: \n{beta}\n\n")
            # print(f"objective_fn: \n{objective_fn(slice(data.rewards), lyap_risks, beta)}\n\n")
            # breakpoint()

            (
                qf_state,
                (qf_loss_value, ent_coef_value),
                key,
            ) = Lyap_SAC_IP.update_critic(
                gamma,
                actor_state,
                qf_state,
                ent_coef_state,
                slice(data.observations),
                slice(data.actions),
                slice(data.next_observations),
                # slice(data.rewards),
                objective_fn(slice(data.rewards), lyap_risks.flatten(), beta), # Changed reward to lyap risk
                slice(data.dones),
                key,
            )
            qf_state = Lyap_SAC_IP.soft_update(tau, qf_state)

            if i in policy_delay_indices:
                (actor_state, qf_state, actor_loss_value, key, entropy) = cls.update_actor(
                    actor_state,
                    qf_state,
                    ent_coef_state,
                    slice(data.observations),
                    key,
                )
                ent_coef_state, _ = Lyap_SAC_IP.update_temperature(target_entropy, ent_coef_state, entropy)
            
            wm_state, wm_loss, wm_info = WorldModel.update(wm_state,
                                                           slice(data.observations), 
                                                           slice(data.actions),
                                                           slice(data.next_observations))

            info: FrozenDict[str, jnp.ndarray] = FrozenDict({"v_candidates mean": v_candidates_mean,
                                                            "lyap_lr": jnp.array(lyap_state.learning_rate),
                                                            "sigma_mean": wm_info["avg sigma"],
                                                            "wm_learning_rate": jnp.array(wm_state.learning_rate)}) # Make adaptive incase of scheduling?


        return (
            qf_state,
            actor_state,
            ent_coef_state,
            lyap_state,
            wm_state,
            key,
            (actor_loss_value, qf_loss_value, ent_coef_value, lyap_loss, wm_loss, info),
        )

