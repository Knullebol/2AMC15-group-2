## Group 2 - Assignment 2

First A2 commit was on the 3rd of June. 


### DQN Agent 
The DQN Agent can be trained and run by navigating to the root directory of the project, and within a terminal running `python "train_dqn.py" --episodes (number of episodes) --steps (number of iterations per episode)`

Additional input parameters: 
- `--logs` : If used the agent will print logs and create plots at the end of training
- `--random_seed (int)` : Random seed value for the environment
- `--location (int)` : Target building to deliver to. 0=Easy, 1=Markthal, 2=Auditorium, 3=Nexus
- `--use_distance` : If used, enables distance based rewards
- `--use_direction` : If used, enables direction based rewards
- `--use_distance` : If used, enables distance based rewards
- `--use_stalling` : If used, enables passive punishments for staying in one area for too long

The hyperparameters of the agent can be found and changed in the `dqn` directory.


### Dueling DQN Agent

The DQN Agent can be trained and run by navigating to the root directory of the project, and within a terminal running `python "train_dueling_double_dqn.py" --episodes (number of episodes) --steps (number of iterations per episode)`

Additional input parameters: 
- `--logs` : If used the agent will print logs and create plots at the end of training
- `--random_seed (int)` : Random seed value for the environment
- `--location (int)` : Target building to deliver to. 0=Easy, 1=Markthal, 2=Auditorium, 3=Nexus
- `--use_distance` : If used, enables distance based rewards
- `--use_direction` : If used, enables direction based rewards
- `--use_distance` : If used, enables distance based rewards
- `--use_stalling` : If used, enables passive punishments for staying in one area for too long

The hyperparameters of the agent can be found and under the `dueling_dqn` directory.


### Experiments 

In order to replicate the results of the experiments, three files were created. 

By running `python :experiment_1_hyperparams.py" --agent_type (dqn / dueling_dqn)`, the experiments on the hyperparameters can be done.
By running `python :experiment_1_rewards.py" --agent_type (dqn / dueling_dqn)`, the experiments on the different rewards systems can be done.
By running `python :experiment_1_locations.py" --agent_type (dqn / dueling_dqn)`, the experiments on the different locations can be done.

These experiments support the use of MLflow (https://mlflow.org/). By running `mlflow ui` in a different terminal, the full experiment setup can be accessed. In the case that the experiments fail due to a faulty URI, simply removing the 'mlruns' directory inside the project and rerunning should fix the issue. 
