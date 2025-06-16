#!/usr/bin/env python3
import gymnasium as gym
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.callbacks import CheckpointCallback
import matplotlib.pyplot as plt
from prob_lyap.utils.wrappers_rd import RandomDelayWrapper, NoneWrapper, AugmentedRandomDelayWrapper, UnseenRandomDelayWrapper
import os
import click

# This should just be a test for when t=1000
LOGGING_SERVICES = ["tb", "wandb", "none"]
DELAY_CLASSES = {"none": [NoneWrapper],
                "unseen": [UnseenRandomDelayWrapper],
                "augmented": [AugmentedRandomDelayWrapper],
                "all": [NoneWrapper, UnseenRandomDelayWrapper, AugmentedRandomDelayWrapper]}

def _get_dir(wrapper: RandomDelayWrapper, env_id: str):
    data_dir = f"{wrapper.__name__}/{env_id}/"
    os.makedirs(data_dir, exist_ok=True)
    model_id = len(os.listdir(data_dir))
    model_name = f"{wrapper.__name__}/{env_id}/TQC_{model_id}"
    return model_name

@click.command()
@click.option("-dt", "--delay-type", default="none", help="Specify delay type: (none | unseen | augmented | all)", show_default=True)
@click.option("-t", "--timesteps", default=80_000, help="Number of training time steps", show_default=True)
@click.option("-lr", "--learning-rate", type=float, default=3e-4, help="Actor learning rate", show_default=True)
@click.option("-qf-lr", "--qf-learning-rate", type=float, default=3e-4, help="Critic learning rate", show_default=True)
@click.option("--progress-bar", default=False, is_flag=True, help="Outputs progress bar for run completion", show_default=True)
@click.option("--verbose", default=False, is_flag=True, help="Prints training stats", show_default=True)
@click.option("--env-id", type=str, default="FetchReach-v2", help="registered gym environment id", show_default=True)
@click.option("--gamma", type=float, default=0.99, help="Discount factor", show_default=True)
@click.option("--buffer-size", type=int, default=1_000_000, help="Replay buffer size", show_default=True)
@click.option("--checkpoint-label", type=str, default="\0", help="save checkpoints with this label")
# @click.option("--log", default="none", type=str, help=f"Specify the name of the logging service ({' | '.join(LOGGING_SERVICES)}) where auto selects the most free GPU", show_default=True)
def pretrain_sac(delay_type, timesteps, learning_rate, qf_learning_rate, progress_bar, verbose, env_id, gamma, buffer_size, checkpoint_label):
    # print(f"Delay type: {delay_type} timesteps: {timesteps}")
    assert delay_type in DELAY_CLASSES.keys(), "Invalid delay type"
    if checkpoint_label!="\0":
        checkpoint_callback = CheckpointCallback(save_freq=500_000, save_path=f"./checkpoints/{checkpoint_label}", name_prefix="SACHER_torch")
    else:
        checkpoint_callback = None

    policy_kwargs = dict(net_arch=[512, 512]) # Increase architecture a bit..
    click.secho("\n".join([f"{k}: {v}" for k,v in vars().items()]), fg="green")

    for wrapper in DELAY_CLASSES[delay_type]:
        click.secho(f"[x] Training {wrapper.__name__}", fg="green")
        model_name = _get_dir(wrapper, env_id)
        env = wrapper(gym.make(env_id))

        model = SAC("MultiInputPolicy", env,
                    learning_rate=learning_rate,
                    verbose=int(verbose),
                    tensorboard_log=f"./logs/torch/{env_id}",
                    # qf_learning_rate=qf_learning_rate,
                    buffer_size=buffer_size,
                    device="cuda",
                    gamma=gamma,
                    replay_buffer_class=HerReplayBuffer,
                    replay_buffer_kwargs=dict(
                    n_sampled_goal=4,
                    goal_selection_strategy="future",
                    ),
                    policy_kwargs=policy_kwargs)
        
        model.learn(total_timesteps=timesteps,
                    log_interval=4,
                    callback=checkpoint_callback,
                     progress_bar=progress_bar)
        model.save(f"pretrained_models/torch/{model_name}")

        del model # To ensure loading works
        # print(f"pretrained_models/{model_name}")
        model = SAC.load(f"pretrained_models/torch/{model_name}")
        click.secho(f"pretrained_models/{model_name}", fg="green")
        print("Creating plot of test episode...")

        env = wrapper(gym.make(env_id, render_mode="human"))
        obs, info = env.reset()
        rewards = []
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated | truncated
            rewards.append(reward)
            if terminated or truncated:
                print(f"Total rewards obtained in this episode: {sum(rewards)}")
                obs, info = env.reset()

        # Plotting the rewards after the loop ends
        plt.plot(rewards)
        plt.xlabel('time steps')
        plt.ylabel('Rewards')
        plt.title('Total Rewards over Episode')
        os.makedirs(f"final_episode_plots/{model_name}", exist_ok=True)

        plt.savefig(f"final_episode_plots/{model_name}.png")
        plt.close()

if __name__ == "__main__":
    pretrain_sac()
