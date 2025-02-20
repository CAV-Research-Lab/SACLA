from functools import partial
import numpy as np
import sbx
import gymnasium as gym

import flax
import flax.linen as nn
from flax.training.train_state import TrainState
from flax.core import FrozenDict

import jax
import jax.numpy as jnp
import jax.random as random

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise

from sbx.common.off_policy_algorithm import OffPolicyAlgorithmJax
from sbx.common.type_aliases import ReplayBufferSamplesNp, RLTrainState
from sbx.sac.policies import SACPolicy

class Lyap_net(nn.Module):
    lyap_n_hidden: int
    lyap_n_layers: int
    
    @nn.compact
    def __call__(self, x): 
        x = nn.tanh(nn.Dense(self.lyap_n_hidden, name="input layer")(x))
        for i in range(self.lyap_n_layers):
           x = nn.tanh(nn.Dense(self.lyap_n_hidden, name=f"hidden layer")(x))
        lyapunov_value = jnp.abs(nn.Dense(1, name="output layer")(x)) # relu made it always predict zero
        return lyapunov_value

    @staticmethod
    @jax.jit
    def lie_derivative(lyap_state: TrainState,
                       wm_state: TrainState,
                       observations: np.ndarray,
                       actions: np.ndarray,
                       v_candidate: jnp.ndarray):

        wm_next_obs_mus, _ = wm_state.apply_fn(wm_state.params, observations, actions) # Use wm instead of off-policy actions - could do only once the model is ready
        next_vs = lyap_state.apply_fn(lyap_state.params, wm_next_obs_mus)

        dv_dx_candidate = next_vs - v_candidate # stepwise derivative using next state instead of prev

        lie_derivative = dv_dx_candidate #* f_value
        return lie_derivative 
        
    @staticmethod
    @jax.jit
    def update(lyap_state: TrainState,
                wm_state: TrainState,
                actor_state: TrainState,
                # actions: jnp.ndarray,
                # achieved_goals: jnp.ndarray,
                # desired_goals: jnp.ndarray,
                observations: jnp.ndarray,
                # next_observations: jnp.ndarray,
                key: jax.Array): # get eq_state from self
        
        observations = jnp.array(observations) # Incase of DEBUG

        def loss_fn(params):
            # C1 = 1 
            C2 = 5
            C3 = 1
            v0_observations = observations.at[:, :3].set(observations[:, 3:6]) # Replace ag with dg
            v0_observations = v0_observations.at[:, 6:9].set(observations[:, 3:6])
            v0 = lyap_state.apply_fn(params, v0_observations)

            new_key, noise_key = jax.random.split(key)
            act_dist = actor_state.apply_fn(actor_state.params, observations)
            targ_actions = act_dist.sample(seed=noise_key)
            v_candidate = lyap_state.apply_fn(params, observations)  
            lie_derivative = Lyap_net.lie_derivative(lyap_state,
                                                     wm_state,
                                                     observations,
                                                     targ_actions,
                                                     v_candidate)

            lyap_risks = C2*jnp.maximum(0, -jnp.sum(lie_derivative) ) +\
                                   C3*jnp.power(v0, 2) 

            return lyap_risks.mean(), (lyap_risks, v_candidate.mean(), new_key)
        
        gradient_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (lyap_loss, (lyap_risks, info, new_key)), dv_dp = gradient_fn(lyap_state.params)
        lyap_state = lyap_state.apply_gradients(grads=dv_dp)
        return lyap_state, lyap_loss, lyap_risks, info, new_key


