# Efficient Neural Lyapunov Function Approximation with Reinforcement Learning

## NOTE
The previous SACLA codebase is being migrated here and therefore there are some files etc missing. We are hoping to finish this by May 2025, however in the meantime feel free to reach out to lm01065@surrey.ac.uk with any questions

Installation instructions
1. clone repo with `https://github.com/CAV-Research-Lab/SACLA.git`
2. Install requirements via `pip install -e .` when in the root folder. This will install all the requirements to build the prob_lyap package from the `pyproject.toml` file.
3. run `main.py --help` to see configuration options.

We recommend using environments from the Gymnasium robotics benchmark or `InvertedPendulum-v4` however you can easily modify the chosen equilibrium state in `lyap_func_InvertedPendulum.py` to extend to other non goal-conditioned environments.

There should be some updates coming soon as we refactor the codebase! If you would like to collaberate or have any suggestions please feel free to reach out.

If you would like to cite this work please use the following:
`@ X`
