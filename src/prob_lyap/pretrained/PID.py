import gymnasium as gym
import numpy as np
# import numba
from functools import partial
import click
from tqdm import tqdm
import jax.numpy as jnp

'''
Best Average Score: -5.0
Best P: 2.5
Best I: 0.3485714285714286
Best D: 0.002


'''

class PID:
    def __init__(self, P: float=0.5, I: float=0.1, D: float=0.01):
        self.P = P
        self.I = I
        self.D = D
        self.reset()
    
    def reset(self):
         self.integral = 0
         self.prev_error = 0

    # @partial(numba.jit, static_argnames=["self"])
    def act(self, error: float):
            self.integral += error
            derivative = (error - self.prev_error).sum()
            self.prev_error = error
            action = self.P*error + self.I*self.integral + self.D*derivative
            return action # jnp.append(action, jnp.array([0]))

@click.command()
@click.option("-n", default=5, type=int, help="The number of numbers to test per param (P,I,D) during grid search", show_default=True)
@click.option("-s", default=5, type=int, help="The number of seeds to average per episode during grid search", show_default=True)

def main(n: int, s:int):
    click.secho("Grid search for PID parameters", fg="yellow")
    best_score = float('-inf')
    best_P = None
    best_I = None
    best_D = None
    env = gym.make('FetchReach-v2')

    num_trials = s  # Number of trials for each PID combination
    # np.linspace(0.1, 2.5, n)
    for P in tqdm([2.5]):
        seed = np.random.randint(999)
        for I in np.linspace(0, 0.5, n):
            for D in np.linspace(0, 0.2, n):
                trial_scores = []
                for i in range(num_trials):
                    controller = PID(P, I, D)
                    obs, _ = env.reset(seed=seed+i)
                    score = 0
                    done = False
                    while not done:
                        error = obs['desired_goal'] - obs['achieved_goal']
                        action = controller.act(error)  # Assuming error is 0 for initial action
                        obs, reward, terminated, truncated, info = env.step(action)
                        done = terminated | truncated
                        score += reward
                    trial_scores.append(score)

                avg_score = sum(trial_scores) / num_trials

                if avg_score > best_score:
                    best_score = avg_score
                    best_P = P
                    best_I = I
                    best_D = D
    env.close()

    print("Best Average Score:", best_score)
    print("Best P:", best_P)
    print("Best I:", best_I)
    print("Best D:", best_D)
    return avg_score

if __name__ == "__main__":
     main()