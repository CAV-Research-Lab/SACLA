import gymnasium as gym
from gymnasium.spaces import Dict
import gymnasium_robotics

from prob_lyap.utils.utils import seed_modules, get_ckpt_dir, get_most_free_gpu
from prob_lyap.lyap_SAC import Lyap_SAC
from prob_lyap.lyap_SAC_invertedpendulum import Lyap_SAC_IP

from pprint import pprint
from copy import deepcopy
import os
from pathlib import Path
import jax

from prob_lyap.utils.type_aliases import LyapConf
import click
from typing import Optional
import ast

DEVICES = ["auto", "cpu", "cuda"]

LOGGING_SERVICES = ["tb", "wandb", "none"]

def run(lyap_config: LyapConf,
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

    # Make environment
    env = gym.make(lyap_config.env_id)
    _, _, env = seed_modules(deepcopy(env), seed=lyap_config.seed)

    # Logging
    lyap_config.ckpt_dir, run_id = get_ckpt_dir(lyap_config) # TODO: Custom paths for ckpts
    run_dir = Path() / lyap_config.env_id 

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
        model = Lyap_SAC_IP(lyap_config, "MultiInputPolicy", deepcopy(env), policy_kwargs=dict(net_arch=actor_net_arch), gamma=gamma, qf_learning_rate=qf_lr, verbose=int(verbose), tensorboard_log=run_folder, device=device)
    else:
        model = Lyap_SAC(lyap_config, "MultiInputPolicy", deepcopy(env), policy_kwargs=dict(net_arch=actor_net_arch), gamma=gamma, qf_learning_rate=qf_lr, verbose=int(verbose), tensorboard_log=run_folder, device=device)

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
@click.option("--progress-bar", default=False, is_flag=True, help="Outputs progress bar for run completion", show_default=True)
@click.option("--verbose", default=False, is_flag=True, help="Show training stats", show_default=True)
@click.option("-s", "--seed", type=int, default=0, help="Random seed", show_default=True)
@click.option("-nh", "--n-hidden", type=int, default=16, help="Number of hidden neurons in Lyapunov neural network", show_default=True)
@click.option("-nl", "--n-layers", type=int, default=1, help="Number of mlp layers in Lyapunov neural network", show_default=True)
@click.option("--gamma", type=float, default=0.99, help="RL discount factor", show_default=True)
@click.option("--lyap-lr", type=float, default=4e-7, help="Lyapunov neural network learning rate", show_default=True)
@click.option("--wm-lr", type=float, default=0.00003, help="World model neural network learning rate", show_default=True)
@click.option("--ckpt-every", type=int, default=20_000, help="Checkpoint saving frequency", show_default=True)
@click.option("--actor-lr", type=float, default=0.0001, help="actor neural network learning rate", show_default=True)
@click.option("--qf-lr", "--qf-learning-rate", type=float, default=3e-4, help="Critic learning rate", show_default=True)
@click.option("--actor-net-arch", default=None, help="actor neural network architecture as list, default is [256, 256]", show_default=True)
@click.option("-t", "--timesteps", default=80_000, type=int, help="training timesteps", show_default=True)
@click.option("--env-id", type=str, default="FetchReachDense-v3", help="registered gym environment id", show_default=True)
@click.option("--ckpt-dir", default="default",type=str, help="Directory to save run checkpoints, by default this saves in ~/.prob_lyap", show_default=True)
@click.option("-b", "--beta", default=0.1, type=float, help="Balancing parameter for SACLA", show_default=True)
@click.option("--debug", default=False, is_flag=True, type=bool, help="Disables jax.jit and provides additional debugging information", show_default=True)
@click.option("--device", default="auto", type=str, help=f"Specify the name of the device to run on ({' | '.join(DEVICES)}) where auto selects the most free GPU", show_default=True)
@click.option("--log", default="tb", type=str, help=f"Specify the name of the logging service ({' | '.join(LOGGING_SERVICES)}) where auto selects the most free GPU", show_default=True)
def main(progress_bar: bool,
         verbose: bool,
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
         ckpt_dir: str,
         beta: float,
         debug: bool,
         device: str,
         log: str):

    assert device in DEVICES, f"Select a device from these options: {DEVICES}"
    assert log in LOGGING_SERVICES, f"{log} is not a valid logging service. Choose from: {LOGGING_SERVICES}"
    
    if device=="auto": 
        import jax
        if any(d.platform == 'gpu' for d in jax.devices()):
            os.environ["CUDA_VISIBLE_DEVICES"] = str(get_most_free_gpu())
            import jax # Has to be done after setting device
    
    click.secho(f"Running SACLA on: {jax.devices()[0]}", fg="green")

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
        ckpt_dir=ckpt_dir, 
        beta=beta,
        debug=debug,
        logging=log,
    )
    
    run(lyap_conf, progress_bar, verbose, actor_net_arch, gamma, qf_lr, device)

if __name__ == "__main__":
    main()

