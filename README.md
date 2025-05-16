Open terminal in the same directory, here are example commands to run:
1. Running value iteration agent:  
	python "train_vi.py" ".\grid_configs\test_grid.npy" --iter=1000 --no_gui --sigma=0.1  --gamma=0.9 --random_seed=0

	Additional option:
		--gamma: Discount factor


2. Running Monte Carlo agent:
	python "train_mc_merged.py" ".\grid_configs\A1_grid.npy" --iter=100000 --no_gui --sigma=0.1  --random_seed=0
	
	Note:
	· Modify max steps: set "max_steps" variable appropriately in train_mc_merged.py 

3. Running Q-Learning agent:
	python "train_q.py" ".\grid_configs\A1_grid.npy" --iter=100000 --no_gui --sigma=0.1  --gamma=0.9  --alpha=0.05  --epsilon=0.3 --random_seed=0 

	Additional option:
		--gamma: Discount factor
		--epsilon: Epsilon for epsilon-greedy action selection.
		--alpha: Learning rate for Q-learning