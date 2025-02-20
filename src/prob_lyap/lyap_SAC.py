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

# from gymnasium.spaces.dict import Dict

from prob_lyap.utils.type_aliases import ReplayBufferSamplesNp, RLTrainState, CustomTrainState, LyapConf
from prob_lyap.utils import utils
from prob_lyap.lyap_func import Lyap_net
from prob_lyap.lyap_func_InvertedPendulum import Lyap_net_IP
from prob_lyap.world_model import WorldModel
from prob_lyap.objectives import get_objective

from stable_baselines3.common.buffers import RolloutBuffer

from copy import deepcopy
from typing import Callable

from typing import Dict
import copy

class Lyap_SAC(sbx.SAC):    
    def __init__(self, lyap_config: LyapConf, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.objective_fn = get_objective(lyap_config.objective)
        self.learning_rate = lyap_config.actor_lr
        self.lyap_config = lyap_config
        self.rng = random.PRNGKey(lyap_config.seed)# Take in as arg or random.PRNGKey(1)
        self.debug = lyap_config.debug
        
        lyap = Lyap_net(lyap_config.n_hidden, lyap_config.n_layers)        

        if isinstance(self.env.observation_space.sample(), Dict): # For fetch envs
            wm = WorldModel(sum([x.shape[0] for x in list(self.env.observation_space.values())]))
        elif lyap_config.env_id=="InvertedPendulum-v4":
            wm = WorldModel(self.env.observation_space.shape[0])
        else:
            raise "Unsupported env"
            
        lyap_key, wm_key, self.rng = random.split(self.rng, num=3)
        self.lyap_state = utils.create_train_state("lyapunov", lyap_key, lyap, lyap_config.lyap_lr, self.env)
        self.wm_state = utils.create_train_state("world model", wm_key, wm, lyap_config.wm_lr, self.env) # For now use the same lr, change in future
        self.ckpt_dir = lyap_config.ckpt_dir
        self.beta = lyap_config.beta

        # Checkpoints
        options = ocp.CheckpointManagerOptions(max_to_keep=None, create=True)
        checkpointer = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())
        self.checkpoint_manager = ocp.CheckpointManager(self.ckpt_dir, checkpointers=checkpointer, options=options)

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
            lyap_state, lyap_loss, lyap_risks, v_candidates_mean, key = Lyap_net.update(lyap_state,
                                                           wm_state,
                                                           actor_state,
                                                           slice(data.observations),
                                                           key)

            (
                qf_state,
                (qf_loss_value, ent_coef_value),
                key,
            ) = Lyap_SAC.update_critic(
                gamma,
                actor_state,
                qf_state,
                ent_coef_state,
                slice(data.observations),
                slice(data.actions),
                slice(data.next_observations),
                objective_fn(slice(data.rewards), lyap_risks.flatten(), beta), # Changed reward to lyap risk
                slice(data.dones),
                key,
            )
            qf_state = Lyap_SAC.soft_update(tau, qf_state)

            if i in policy_delay_indices:
                (actor_state, qf_state, actor_loss_value, key, entropy) = cls.update_actor(
                    actor_state,
                    qf_state,
                    ent_coef_state,
                    slice(data.observations),
                    key,
                )
                ent_coef_state, _ = Lyap_SAC.update_temperature(target_entropy, ent_coef_state, entropy)
            
            wm_state, wm_loss, wm_info = WorldModel.update(wm_state,
                                                           slice(data.observations), 
                                                           slice(data.actions),
                                                           slice(data.next_observations))


            info: FrozenDict[str, jnp.ndarray] = FrozenDict({"v_candidates mean": v_candidates_mean,
                                                            "lyap_lr": jnp.array(lyap_state.learning_rate),
                                                            "sigma_mean": wm_info["avg sigma"],
                                                            "mu_mean": wm_info["avg mu"],
                                                            "wm_abs_loss": wm_info["abs_diff"],
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


    def train(self, batch_size, gradient_steps):
        data = self.replay_buffer.sample(batch_size * gradient_steps, env=self._vec_normalize_env)
        policy_delay_indices = {i: True for i in range(gradient_steps) if ((self._n_updates + i + 1) % self.policy_delay) == 0}
        policy_delay_indices = flax.core.FrozenDict(policy_delay_indices)

        if isinstance(data.observations, Dict):
            keys = list(self.observation_space.keys())
            obs = np.concatenate([data.observations[key].numpy() for key in keys], axis=1) # Perhaps modify this for the loss thingy? Make a froxen dict?
            next_obs = np.concatenate([data.next_observations[key].numpy() for key in keys], axis=1)
            desired_goals = data.observations['desired_goal'].numpy()
            achieved_goals = data.observations['achieved_goal'].numpy()
            next_desired_goals = data.next_observations['desired_goal'].numpy()
            next_achieved_goals = data.next_observations['achieved_goal'].numpy()
        else:
            obs = data.observations.numpy()
            next_obs = data.next_observations.numpy()
            desired_goals = np.zeros_like(obs)
            achieved_goals = deepcopy(obs)
            next_desired_goals = np.zeros_like(next_obs)
            next_achieved_goals = deepcopy(next_obs)

        # Convert to numpy
        data = ReplayBufferSamplesNp(
            obs,
            data.actions.numpy(),
            next_obs,
            data.dones.numpy().flatten(),
            data.rewards.numpy().flatten(),
            achieved_goals,
            desired_goals,
            next_achieved_goals,
            next_desired_goals,
        )
        (
            self.policy.qf_state,
            self.policy.actor_state,
            self.ent_coef_state,
            self.lyap_state,
            self.wm_state,
            self.key,
            (actor_loss_value, qf_loss_value, ent_coef_value, lyap_loss, wm_loss, info),
        ) = self._train(
            self.gamma,
            self.tau,
            self.target_entropy,
            gradient_steps,
            data,
            policy_delay_indices,
            self.policy.qf_state,
            self.policy.actor_state,
            self.ent_coef_state,
            self.lyap_state, #
            self.wm_state, #
            self.key, #
            self.objective_fn, #
            self.beta, # 
            self.debug
        )
        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/actor_loss", actor_loss_value.item())
        self.logger.record("train/critic_loss", qf_loss_value.item())
        self.logger.record("train/ent_coef", ent_coef_value.item())
        self.logger.record("train/lyap_loss", lyap_loss.item())
        self.logger.record("train/wm_loss", wm_loss.item())

        # Info: FrozenDict of info for logging
        for key, value in info.items():
            try:
                self.logger.record(f"train/{key}", value.item()) # .item()
            except:
                breakpoint()

        # Checkpoint lyap_state 
        if self.lyap_config.logging != "none" and not self.lyap_config.debug and (self.num_timesteps % self.lyap_config.ckpt_every == 0):
            self.save_state()

    def save_state(self) -> None:
        ckpt = {'lyap_state': self.lyap_state,
                 'config': self.lyap_config.__str__(),
                 'actor_state': self.policy.actor_state,
                 'wm_state': self.wm_state}

        save_args = orbax_utils.save_args_from_target(ckpt)
        self.checkpoint_manager.save(self.num_timesteps, ckpt, save_kwargs={'save_args': save_args})

    def restore_ckpt(self, step:int) -> TrainState:
        if not step: # Get latest if not specified
            step = self.checkpoint_manager.latest_step()
        assert step, f"Step should be an integer. Step: {step}"
        click.secho(f"Loading step: {step}", fg="green")
        restored_ckpt = self.checkpoint_manager.restore(step)
        self.lyap_state = CustomTrainState(**restored_ckpt['lyap_state'], apply_fn=self.lyap_state.apply_fn, tx=None)
        self.lyap_config = restored_ckpt['config']
        self.policy.actor_state = TrainState(**restored_ckpt["actor_state"], apply_fn=self.policy.actor_state.apply_fn, tx=None)
        self.lyap_config = restored_ckpt['config']
        self.wm_state = CustomTrainState(**restored_ckpt['wm_state'], apply_fn=self.wm_state.apply_fn, tx=None)

        # Assert actor params are now equal (Default vs Loaded)
        # assert not utils.params_equal(prev_actor_params['params'], self.policy.actor_state.params['params']), "Updated params same as default"
        # assert utils.params_equal(restored_ckpt['actor_state']['params']['params'], self.policy.actor_state.params['params']), "Updated params different to loaded params"

        # # Assert lyap params are now equal (Default vs Loaded)
        # assert not utils.params_equal(prev_lyap_params['params'], self.lyap_state.params['params']), "Updated params same as default"
        # assert utils.params_equal(restored_ckpt['lyap_state']['params']['params'], self.lyap_state.params['params']), "Updated params different to loaded params"

        # # Assert wm params are now equal (Default vs Loaded)
        # assert not utils.params_equal(prev_wm_params['params'], self.wm_state.params['params']), "Updated params same as default"
        # assert utils.params_equal(restored_ckpt['wm_state']['params']['params'], self.wm_state.params['params']), "Updated params different to loaded params"

        return restored_ckpt
