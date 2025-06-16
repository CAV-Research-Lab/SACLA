import numpy as np

import flax.linen as nn
from flax.training.train_state import TrainState

import jax
import jax.numpy as jnp



class Lyap_net_IP(nn.Module):
    lyap_n_hidden: int
    lyap_n_layers: int
    # Make sure this is getting jitted
    @nn.compact
    def __call__(self, x): # Should be a function of error?
        x = nn.tanh(nn.Dense(self.lyap_n_hidden, name="input layer")(x))
        for i in range(self.lyap_n_layers):
            # x = nn.tanh(nn.Dense(self.lyap_n_hidden, name=f"hidden layer_{i}")(x))
           x = nn.tanh(nn.Dense(self.lyap_n_hidden, name="hidden layer")(x))
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

        error_value = jnp.abs(jnp.abs(observations[1])) # Make sure these indexes are correct
        error_value_next = jnp.abs(wm_next_obs_mus[1])
        # error_value_next = jnp.abs(wm_next_obs_mus[:, -6:-3] - wm_next_obs_mus[:, -3:])

        dv_dx_candidate = next_vs - v_candidate # stepwise derivative using next state instead of prev
        f_value = error_value_next - error_value

        lie_derivative = dv_dx_candidate * f_value
        return lie_derivative 
        
        # dvdx decreasing -ive, f_value decreasing -ive = +ive LfV(x)
        # lie_derivative = lie_derivative.reshape((error.shape[0], -1)) # Lie derivative as vector
        # f_value = jnp.abs(wm_next_ag_mus - wm_next_dg_mus) ### !!! nah
        # error = jnp.sum(f_value, axis=1).reshape(-1,1,1)
        #f_value = jnp.abs(wm_next_obs_mus[:, -6:-3] - wm_next_obs_mus[:, -3:]) # This isn't gonna jit and should be adaptive anyway (just ag)


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
            C1 = 1 # Emphasise certain conditions
            C2 = 5
            C3 = 1
            v0_observations = observations.at[:, 1].set(0) # Replace ag with dg
            v0_observations = v0_observations.at[:, 1].set(0)
            v0 = lyap_state.apply_fn(params, v0_observations)
            ###
            # v0 = lyap_state.apply_fn(lyap_state.params, jnp.zeros_like(observations)) # state.params or params? lax stop grad?
            # dv_dx = jax.grad(lambda obs: lyap_state.apply_fn(params, obs).mean()) # .mean()? grad wrt x
            # dv_dx_candidate = dv_dx(observations)
            new_key, noise_key = jax.random.split(key)
            act_dist = actor_state.apply_fn(actor_state.params, observations)
            targ_actions = act_dist.sample(seed=noise_key)
            v_candidate = lyap_state.apply_fn(params, observations)  
            lie_derivative = Lyap_net_IP.lie_derivative(lyap_state,
                                                     wm_state,
                                                     observations,
                                                     targ_actions,
                                                     v_candidate)

            lyap_risks = C2*jnp.maximum(0, -jnp.sum(lie_derivative) ) +\
                                   C3*jnp.power(v0, 2) # Try with +ive Lie^

            return lyap_risks.mean(), (lyap_risks, v_candidate.mean(), new_key)
        
        gradient_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (lyap_loss, (lyap_risks, info, new_key)), dv_dp = gradient_fn(lyap_state.params) # gradient wrt params
        lyap_state = lyap_state.apply_gradients(grads=dv_dp)
        return lyap_state, lyap_loss, lyap_risks, info, new_key

