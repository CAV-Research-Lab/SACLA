import jax
import jax.numpy as jnp
from jax import random
import flax
import flax.linen as nn
from flax.training.train_state import TrainState
import distrax
import gymnasium as gym
import numpy as np
# from prob_lyap.lyap_func import Lyap_net

class WorldModel(nn.Module):
    num_outputs: int
    probabalistic: bool = False

    @nn.compact
    def __call__(self, x, a):
        x = jnp.concatenate([x, a], -1)
        x = nn.relu(nn.Dense(16, name="Input Layer")(x))
        mus = nn.Dense(self.num_outputs, name="Mu Layer")(x)
        sigmas = nn.softplus( nn.Dense(self.num_outputs, name="Sigma Layer")(x) )
        return mus, sigmas

    @staticmethod
    @jax.jit
    def update(wm_state: TrainState, observations: np.ndarray,
                actions: np.ndarray, next_observations: np.ndarray):
            
            def loss_fn(params: flax.core.FrozenDict):
                mus, sigmas = wm_state.apply_fn(params, observations, actions)
                pred_diff = jax.lax.stop_gradient(abs(mus - next_observations).mean()) # Metric to ensure wm get's better
                state_distribution = distrax.MultivariateNormalDiag(mus, sigmas) # Assume invariance
                pred_loss = -state_distribution.log_prob(next_observations) # Redefine Lyap Risk
                info = flax.core.FrozenDict({"avg mu": mus.mean(),
                                            "avg sigma": sigmas.mean(),
                                            "abs_diff": pred_diff})
                return pred_loss.mean(), info
            
            gradient_fn = jax.value_and_grad(loss_fn, has_aux=True)
            (loss, info), dv_dp = gradient_fn(wm_state.params) # gradient wrt params
            wm_state = wm_state.apply_gradients(grads=dv_dp)
            return wm_state, loss, info

def flatten_obs(obs: dict) -> np.ndarray:
    output = np.array([])

    for arr in obs.values():
        output = np.append(output, arr)

        return output


if __name__ == "__main__":
    seed = 0
    
    rng = random.PRNGKey(seed)
    env = gym.make("FetchPush-v2")
    obs, _ = env.reset(seed=seed)
    obs_size = len( jnp.hstack([x for x in obs.values()]) )
    wm  = WorldModel(obs_size=obs_size)
    wm_key, rng = random.split(rng)
    learning_rate = 0.0001

    sample_obs = jnp.hstack([x for x in obs.values()]).reshape(1,-1)
    sample_act = jnp.array(env.action_space.sample()).reshape(1,-1)

    wm_state = wm.create_train_state(wm_key, wm, learning_rate, env)

    mus, sigmas = wm_state.apply_fn(wm_state.params, sample_obs, sample_act)
    print(f"mus: {mus}\nsigmas:{sigmas}")

