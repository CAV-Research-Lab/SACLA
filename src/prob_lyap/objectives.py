
import jax
import jax.numpy as jnp

# '''
# TODO:
#  - Generate plots comparing POLYC to LSAC
#  - Generate plots comparing adverserial, polyc, and standard for each env
#  - Extend to FetchPush

#  - Add to paper how we can use this to analyse RL algos, i.e. on PID Lyap function get's 99% but on RL it gets 45% shows RL is worse, also used to compare between RLs
#  - Ensure all objectives are implemented correctly, signs + clipping?)
# Mention in paper that we use our definition of the Lie Derivative for consistancyand compatability within the goal-conditioned setting when using POLYC!


#   Use tilde or hat? Ask GPT maybe
#  loss_{\hat{f}}(\theta, x) = -\Simga^T_{t=1}log\hat{f}_{\theta}(x_{t+1}|x_t,a_t)
#  Our probabalistic model models a Guassian distribution with 
#  diagonal covariances, parametrized by $\theta$ and conditioned on $x-t$ and $a_t$, 
#  i.e.: $\hat{f}=\p(x_{t+1}|x_t,a_t) = \mathcal{N}(\mu_{\theta}(x_t,a_t),\sigma_{\theta}(x_t,a_t) )$ % check letters are right
#  then the loss becomes:
#  \begin{equation}
#     loss_{\hat{f}}(\theta)=\sigma^T_t[\mu_{\theta}(x_t,a_t)-x_{t+1}]^T\sigma^{-1}_{\theta}(x_t,a_t)[\mu_{theta}(x_t,a_t)-x_{t+1}]+\texttt{log det}\sigma_{\theta}(x_t,a_t). 
#  \end{equation}
 
#  aleotoric uncertanty = heteroscedastic noise
#  Guassian is reasonable if we assume stochasticity is uni-modal
#  However, any tractable distribution can be used
#  To provide for an expressive dynamics model, we can represent the parameters of this distribution (e.g., the mean
# and covariance of a Gaussian) as nonlinear, parametric functions of the current state and action,
# which can be arbitrarily complex but deterministic
# Trade-off is inaccuracy caused from ill-define variance for out of distribution transitions which is why we employ
# the clipped gradeint trick from PETS

# IMPLEMENT THE CLIPPING TRICK FROM APPENDIX A1 TO REDUCE THE VARIANCE
#  '''

@jax.jit
def standard(rewards: jnp.ndarray, lyap_risks: jnp.ndarray, beta: float):
    return rewards#.flatten()

@jax.jit
def adverserial(rewards: jnp.ndarray, lyap_risks: jnp.ndarray, beta: float):
    return -lyap_risks#.flatten() # -ive because we wan't to maximise this

@jax.jit
def POLYC_objective(rewards: jnp.ndarray, lyap_risks: jnp.ndarray, beta: float):
    # Divide risks by 100 in order to fix scaling with reward? # Why clip again?
    # return (beta * jnp.clip(lyap_risks, a_max=0)) + ( (1-beta) * rewards ) # Try not flatten
    return (beta * lyap_risks) + ( (1-beta) * rewards ) # Try not flatten

## REMOVED CLIP FROM LYAP RISKS
@jax.jit
def mixed_adv_objective(rewards: jnp.ndarray, lyap_risks: jnp.ndarray, beta: float):
    # Inverse POLYC
    return (beta * -lyap_risks + ((1-beta) * rewards )) # -ve in order to maximise, normalisation?

OBJ_TYPES = {"standard": standard,
            "adverserial": adverserial,
            "polyc": POLYC_objective,
             "mixed_adv": mixed_adv_objective}

def get_objective(objective="standard") -> None:
    return OBJ_TYPES[objective]
