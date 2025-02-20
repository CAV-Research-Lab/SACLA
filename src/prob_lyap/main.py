import gymnasium as gym
from gymnasium.spaces import Dict
from gymnasium.wrappers.flatten_observation import FlattenObservation

from prob_lyap.utils.wrappers_rd import NoneWrapper, UnseenRandomDelayWrapper, AugmentedRandomDelayWrapper
from prob_lyap.utils.utils import seed_modules, get_ckpt_dir, get_most_free_gpu
from prob_lyap.lyap_SAC import Lyap_SAC
from prob_lyap.lyap_SAC_invertedpendulum import Lyap_SAC_IP
from prob_lyap.lyap_func_InvertedPendulum import Lyap_net_IP
from prob_lyap.objectives import OBJ_TYPES
from prob_lyap.polyc.POLYC import POLYC

from pprint import pprint
from copy import deepcopy
import os
import jax.numpy as jnp
from pathlib import Path
import jax

from prob_lyap.utils.type_aliases import LyapConf
import click
from typing import Literal, Optional
from enum import Enum
import ast

DEVICES = ["auto", "cpu", "cuda"]

DELAY_CLASSES = {"none": NoneWrapper,
                "unseen": UnseenRandomDelayWrapper,
                "augmented": AugmentedRandomDelayWrapper}

LOGGING_SERVICES = ["tb", "wandb", "none"]

ALGOS = ["polyc", "lsac"]

def run(run_all_delay_types:bool,
        run_all_objectives: bool,
        algorithm: str,
        lyap_config: LyapConf,
        progress_bar: bool=False,
        verbose: bool=False,
        actor_net_arch=None,
        gamma: float=0.99,
        qf_lr: float = 3e-4,
        device: str = "auto"):
    
    if actor_net_arch:
        actor_net_arch = ast.literal_eval(actor_net_arch)
        print("Net arch:", actor_net_arch)
    callbacks = None
    # Batch runs

    if run_all_delay_types:
        delay_types = [AugmentedRandomDelayWrapper, UnseenRandomDelayWrapper, NoneWrapper]
        click.secho(f"Starting batch run with {', '.join([x.__name__ for x in delay_types])} delay types", fg="green")
    else:
        delay_types = [lyap_config.delay_type]

    if run_all_objectives:
        obj_types = list(OBJ_TYPES.keys())
        click.secho(f"Starting batch run with {', '.join(obj_types)} objectives", fg="green")

    else:
        obj_types = [lyap_config.objective]

    # Main training
    
    for delay_type in delay_types:
        for obj_type in obj_types:
            lyap_config.objective = obj_type
            lyap_config.delay_type = delay_type

            # Make environment
            env = lyap_config.delay_type( gym.make(lyap_config.env_id), act_delay_range=lyap_config.act_delay, obs_delay_range=lyap_config.obs_delay )
            _, _, env = seed_modules(deepcopy(env), seed=lyap_config.seed)

            # Logging
            lyap_config.ckpt_dir, run_id = get_ckpt_dir(lyap_config, algorithm=algorithm) # TODO: Custom paths for ckpts
            run_dir = Path() / lyap_config.env_id / lyap_config.objective / lyap_config.delay_type.__name__ 

            # if lyap_config.logging == "tensorboard":
            run_folder = Path(__file__).parent / "logs" / run_dir

            if lyap_config.logging == "wandb":
                import wandb
                from wandb.integration.sb3 import WandbCallback

                run = wandb.init(
                    project="LyapRL",
                    name=f"{run_dir}",
                    config=lyap_config.__dict__,
                    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
                    monitor_gym=True,  # auto-upload the videos of agents playing the game
                    save_code=True,  # optional
                )
                # run_folder = None
                callbacks = WandbCallback(gradient_save_freq=100,
                                          model_save_path=f"wandb/wb_models/{run_dir / run.id}",
                                          verbose=2)
                
            elif lyap_config.logging == "tb":
                run_folder = run_folder / str(run_id)
            
            elif lyap_config.logging == "none":
                run_folder = None
                click.secho("Not logging or saving checkpoints", fg="green")
            
            click.secho(f"Log dir: {run_folder}", fg="green")
            if lyap_config.env_id == "InvertedPendulum-v4":

                if algorithm == "polyc":
                    lyap_config.objective = "polyc"
                    model = POLYC(lyap_config, "MlpPolicy", FlattenObservation(deepcopy(env)), verbose=int(verbose), tensorboard_log=run_folder, device=device) 
                elif algorithm == "lsac": # Actor LR is added throguh the config
                    model = Lyap_SAC_IP(lyap_config, "MultiInputPolicy", deepcopy(env), policy_kwargs=dict(net_arch=actor_net_arch), gamma=gamma, qf_learning_rate=qf_lr, verbose=int(verbose), tensorboard_log=run_folder, device=device)
                else:
                    raise f"Please select a valid algorithm name from ({'|'.join(ALGOS)})"

            else:
                if algorithm == "polyc":
                    lyap_config.objective = "polyc"
                    model = POLYC(lyap_config, "MlpPolicy", FlattenObservation(deepcopy(env)), verbose=int(verbose), tensorboard_log=run_folder, device=device) 
                elif algorithm == "lsac": # Actor LR is added throguh the config
                    model = Lyap_SAC(lyap_config, "MultiInputPolicy", deepcopy(env), policy_kwargs=dict(net_arch=actor_net_arch), gamma=gamma, qf_learning_rate=qf_lr, verbose=int(verbose), tensorboard_log=run_folder, device=device)
                else:
                    raise f"Please select a valid algorithm name from ({'|'.join(ALGOS)})"

            # Printing
            click.secho(f"{'-'*22} CONFIG {'-'*22}", fg="yellow")
            pprint(lyap_config)
            if isinstance(env.observation_space, Dict):
                print(f"Dict Observation Size: {env.observation_space['observation'].shape}\n")
            else:
                print(f"{type(env.observation_space)} Observation Size: {env.observation_space.shape}\n")
            click.secho("-"*54, fg="yellow")

            if not lyap_config.debug:
                model.learn(total_timesteps=lyap_config.total_timesteps, log_interval=4, progress_bar=progress_bar, callback=callbacks)
            else:
                click.secho("DEBUG mode enabled", fg="green")
                with jax.disable_jit():
                    model.learn(total_timesteps=lyap_config.total_timesteps, log_interval=4, progress_bar=progress_bar)
            
            if lyap_config.logging=="tb":click.secho(f"Complete logs at: {run_folder}", fg="green")

        # backup_state(model)
        # model.save_state(run_folder.replace("logs", "RL_models"))

# Add some of these to tensorboard
@click.command()
@click.option("-a", "--algorithm", type=str, default="LSAC", help=f"Which algorithm to train from ({' | '.join(ALGOS)})", show_default=True)
@click.option("--progress-bar", default=False, is_flag=True, help="Outputs progress bar for run completion", show_default=True)
@click.option("--verbose", default=False, is_flag=True, help="Show training stats", show_default=True)
@click.option("--run-all-delays", default=False, is_flag=True, help="Run all delay types", show_default=True)
@click.option("--run-all-objectives", default=False, is_flag=True, help="Run all objective types", show_default=True)
@click.option("-s", "--seed", type=int, default=0, help="Random seed", show_default=True)
@click.option("-nh", "--n-hidden", type=int, default=32, help="Number of hidden neurons in Lyapunov neural network", show_default=True)
@click.option("-nl", "--n-layers", type=int, default=1, help="Number of hidden mlp layers in Lyapunov neural network", show_default=True)
@click.option("--gamma", type=float, default=0.99, help="RL discount factor", show_default=True)
@click.option("--lyap-lr", type=float, default=5e-7, help="Lyapunov neural network learning rate", show_default=True)
@click.option("--wm-lr", type=float, default=0.00003, help="World model neural network learning rate", show_default=True)
@click.option("--ckpt-every", type=int, default=20_000, help="Checkpoint saving frequency", show_default=True)
@click.option("--actor-lr", type=float, default=0.0001, help="actor neural network learning rate", show_default=True)
@click.option("--qf-lr", "--qf-learning-rate", type=float, default=3e-4, help="Critic learning rate", show_default=True)
@click.option("--actor-net-arch", default=None, help="actor neural network architecture as list, default is [256, 256]", show_default=True)
@click.option("-t", "--timesteps", default=80_000, type=int, help="training timesteps", show_default=True)
@click.option("--env-id", type=str, default="FetchReach-v2", help="registered gym environment id", show_default=True)
@click.option("-dt", "--delay-type", default="none", help=f"Specify delay type: ({' | '.join(DELAY_CLASSES.keys())})", show_default=True)
@click.option("--dmin", default=0, type=int, help="The minimum delay (bidirectional, for both observation and action delay)", show_default=True)
@click.option("--dmax", default=4, type=int, help="The maximum delay (bidirectional, for both observation and action delay)", show_default=True)
@click.option("--ckpt-dir", default="default",type=str, help="Directory to save run checkpoints, by default this saves in ~/.prob_lyap", show_default=True)
@click.option("-o", "--objective", type=str, default="standard", help=f"Learning objective function for RL ({' | '.join(OBJ_TYPES)})", show_default=True)
@click.option("-b", "--beta", default=0.5, type=float, help="Balancing parameter for POLYC objective (has no effect on standard or adverserial objectives)", show_default=True)
@click.option("--debug", default=False, is_flag=True, type=bool, help="Disables jax.jit and provides additional debugging information", show_default=True)
@click.option("--device", default="auto", type=str, help=f"Specify the name of the device to run on ({' | '.join(DEVICES)}) where auto selects the most free GPU", show_default=True)
@click.option("--log", default="tb", type=str, help=f"Specify the name of the logging service ({' | '.join(LOGGING_SERVICES)}) where auto selects the most free GPU", show_default=True)
def main(algorithm: str,
         progress_bar: bool,
         verbose: bool,
         run_all_delays: bool,
         run_all_objectives: bool,         
         seed: int,
         n_hidden: int,
         n_layers: int,
         gamma: float,
         lyap_lr: float,
         wm_lr: float,
         actor_lr: float,
         qf_lr: float,
         actor_net_arch: Optional[str],
         ckpt_every: int,
         timesteps: int,
         env_id: str,
         delay_type: str,
         dmin: int,
         dmax: int,
         ckpt_dir: str,
         objective: str,
         beta: float,
         debug: bool,
         device: str,
         log: str):

    algorithm = algorithm.lower()
    objective = objective.lower()
    assert algorithm in ALGOS, f"Algorithm must be one of {ALGOS}"
    assert objective in OBJ_TYPES, f"Objective must be one of {list(OBJ_TYPES.keys())}"
    assert device in DEVICES, f"Select a device from these options: {DEVICES}"
    assert log in LOGGING_SERVICES, f"{log} is not a valid logging service. Choose from: {LOGGING_SERVICES}"
    
    if device=="auto": 
        os.environ["CUDA_VISIBLE_DEVICES"] = str(get_most_free_gpu())
        import jax # Has to be done after setting device
    
    click.secho(f"Running {algorithm.upper()} on: {jax.devices()[0]}", fg="green")

    lyap_conf = LyapConf(
        seed=seed,
        n_hidden=n_hidden,
        n_layers=n_layers,
        lyap_lr=lyap_lr,
        wm_lr=wm_lr,
        actor_lr=actor_lr,
        ckpt_every=ckpt_every,
        total_timesteps=timesteps,
        env_id=env_id, # The environment id i.e. InvertedPendulum-v1 etc.
        delay_type=DELAY_CLASSES[delay_type], # Delay type
        act_delay=range(dmin, dmax), # Action delay range
        obs_delay=range(dmin, dmax), # Observation delay range
        ckpt_dir=ckpt_dir, 
        objective=objective, # TODO: Add suppoort for polyc
        beta=beta,
        debug=debug,
        logging=log,
        # alg_name=alg_name
    )
    ## TESTING 
    # from flax.core import FrozenDict
    # new_conf = jax.tree_map(lambda x: str(x), lyap_conf.__dict__)
    # print(FrozenDict(new_conf))
    run(run_all_delays, run_all_objectives, algorithm, lyap_conf, progress_bar, verbose, actor_net_arch, gamma, qf_lr, device)

if __name__ == "__main__":
    main()
