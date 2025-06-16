import jax.numpy as jnp
from flax.core import FrozenDict
import numpy as np
import gymnasium as gym
import random
import os
from pathlib import Path
from typing import Optional
import GPUtil
import copy
# from prob_lyap.lyap_SAC import Lyap_SAC # Maybe unsafe from circular import
from prob_lyap.utils.type_aliases import LyapConf, CustomTrainState
from prob_lyap.lyap_func import Lyap_net
from prob_lyap.lyap_func_InvertedPendulum import Lyap_net_IP
from typing import Literal
from gymnasium import spaces
import optax
import jax

def create_train_state(model_name: Literal['world model', 'lyapunov'], rng, model, learning_rate, env):
    if isinstance(env.observation_space, spaces.Dict):
        n_obs = sum([x.shape[0] for x in list(env.observation_space.values())])
        update = Lyap_net.update
    else:
        n_obs = env.observation_space.shape[0]
        update = Lyap_net_IP.update

    if model_name == "lyapunov":
        params = model.init(rng, jnp.zeros((1,n_obs)))

    elif model_name == "world model":
        n_act = env.action_space.shape[0]
        params = model.init(rng, jnp.ones((1, n_obs)), jnp.ones((1, n_act)))

    tx = optax.adam(learning_rate)
    return CustomTrainState.create(apply_fn=jax.jit(model.apply), params=params, tx=tx, learning_rate=learning_rate)

def get_most_free_gpu():
    GPUs = GPUtil.getGPUs()
    if not GPUs:
        print("No GPU found, using CPU")
        return "cpu"
    # Sort GPUs by free memory
    GPUs.sort(key=lambda x: x.memoryFree, reverse=True)
    # Return the ID of the GPU with the most free memory
    # breakpoint()
    return GPUs[0].id

def update_config(conf, test_config, run_id):
    conf.update(test_config) # Update the keys we care about
    ckpt_dir, next_run_id = get_ckpt_dir(conf, create=False) # TODO: Change so this isn't through config
    if run_id == -1:
        run_id = next_run_id-1
    conf.ckpt_dir = ckpt_dir.parent / str(run_id)
    return conf


# def flatten_obs(obs: dict):
    # return prepare_obs(obs)
    # return np.concatenate(list(reversed(obs.values()))).reshape(1, -1)
# jnp.concatenate([x for x in obs.values()]).reshape(1, -1)

def params_equal(prev_params: FrozenDict, current_params: FrozenDict):
    assert prev_params.keys() == current_params.keys(), f"param keys are different {prev_params.keys()} vs {current_params.keys()}"
    
    for key in prev_params.keys():

        try:
            if key == "log_std":
                if jnp.array_equal(prev_params[key], current_params[key]):
                    return True

            elif jnp.array_equal(prev_params[key]["kernel"], current_params[key]["kernel"]):
                return True
        except Exception as e:
            print("Exception", e)
            breakpoint()
    else:
        return False

def seed_modules(env: gym.Env, seed: int):
    env.action_space.seed(seed)
    env.unwrapped.action_space.seed(seed)

    random.seed(seed)
    np.random.seed(seed)
    obs, reset_info = env.reset(seed=seed)
    return obs, reset_info, env

def get_model_dir(use_test_dir: bool, test_conf: LyapConf, beta: float, run_id: int, algorithm: str):
    if use_test_dir:
        ckpt_dir = Path(__file__).parent / "models" 
    else:
        ckpt_dir = get_ckpt_dir(test_conf, create=False, algorithm=algorithm)[0].parent

    # handle default cases
    if run_id == -1:
        run_id = np.array(os.listdir(ckpt_dir), dtype=int).max() 
    ckpt_dir = ckpt_dir / str(run_id)

    assert os.path.isdir(ckpt_dir), f"ckpt dir {ckpt_dir} doesnt exist."
    return ckpt_dir


def get_ckpt_dir(lyap_config: LyapConf, create=True):
    '''
    Creates the default checkpoint directory from the provided config
    '''
    base_dir = Path.home() / ".prob_lyap" / "ckpts"        
    ckpt_dir = base_dir / lyap_config.env_id 
    if create: ckpt_dir.mkdir(parents=True, exist_ok=True)    
    run_id = len(os.listdir(ckpt_dir))
    ckpt_dir = ckpt_dir / str(run_id)
    if create: ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir, run_id

def test_episode(policy, env, seed=0):
    obs, _ = env.reset(seed=seed)
    score = 0
    done = False
    while not done:

        try:
            action = policy(obs)
            obs, reward, term, trun, info = env.step(action)
        except:
            import traceback
            print(traceback.print_exc())
            breakpoint()

        score += reward
        done = term | trun

    print("Score:", score)


def get_from_checkpoint(step: Optional[int] = None, model = None):
    '''
    Get's the latest actor checkpoint at specified training step, if step=None gets latest.
    Updates the actor state using this checkpoint and returns the model
    '''
    prev_actor_params = copy.deepcopy(model.policy.actor_state.params)
    prev_lyap_params = copy.deepcopy(model.lyap_state.params)
    prev_wm_params = copy.deepcopy(model.wm_state.params)

    ckpt = model.restore_ckpt(step)  # Get latest checkpoint

    # Assert params start out different (Default vs Loaded)
    assert not params_equal(prev_actor_params['params'], ckpt['actor_state']['params']['params']), "Saved actor params same as default"
    assert not params_equal(prev_lyap_params['params'], ckpt['lyap_state']['params']['params']), "Saved lyap params same as default"
    assert not params_equal(prev_wm_params['params'], ckpt['wm_state']['params']['params']), "Saved wm params same as default"

    # Update the params
    model.policy.actor_state = model.policy.actor_state.replace(params=ckpt['actor_state']['params']) # Update actor params
    model.lyap_state = model.lyap_state.replace(params=ckpt['lyap_state']['params']) # Update lyap params
    model.wm_state = model.wm_state.replace(params=ckpt['wm_state']['params']) # Update lyap params
    
    # Assert actor params are now equal (Default vs Loaded)
    assert not params_equal(prev_actor_params['params'], model.policy.actor_state.params['params']), "Updated params same as default"
    assert params_equal(ckpt['actor_state']['params']['params'], model.policy.actor_state.params['params']), "Updated params different to loaded params"

    # Assert lyap params are now equal (Default vs Loaded)
    assert not params_equal(prev_lyap_params['params'], model.lyap_state.params['params']), "Updated params same as default"
    assert params_equal(ckpt['lyap_state']['params']['params'], model.lyap_state.params['params']), "Updated params different to loaded params"

    # Assert wm params are now equal (Default vs Loaded)
    assert not params_equal(prev_wm_params['params'], model.wm_state.params['params']), "Updated params same as default"
    assert params_equal(ckpt['wm_state']['params']['params'], model.wm_state.params['params']), "Updated params different to loaded params"


    model.lyap_config = ckpt["config"]

    return model

