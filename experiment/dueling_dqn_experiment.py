from experiment.base_experiment import BaseExperiment, ExperimentConfig
from dueling_dqn.dueling_double_dqn import DuelingDoubleDQNAgent
from environment.gym import TUeMapEnv


class DuelingDQNExperiment(BaseExperiment):
    def create_agent(self, state_dim: int, action_dim: int):
        return DuelingDoubleDQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=self.config.hidden_dim,
            lr=self.config.lr,
            gamma=self.config.gamma,
            batch_size=self.config.batch_size,
            buffer_capacity=self.config.memory_size,
            target_update_freq=self.config.target_sync_freq,
            dueling_type='average',
            seed=self.config.seed,
            device=self.device
        )

    def create_environment(self):
        return TUeMapEnv(
            detect_range=self.config.detect_range,
            goal_threshold=self.config.goal_threshold,
            max_steps=self.config.max_steps,
            use_distance=self.config.use_distance,
            use_direction=self.config.use_direction,
            use_stalling=self.config.use_stalling,
            destination=self.config.destination
        )


def run_dueling_dqn_experiment(
    batch_size: int = 64,
    memory_size: int = 10000,
    lr: float = 0.001,
    detect_range: int = 4,
    episodes: int = 500,
    max_steps: int = 200,
    seed: int = 42,
    name: str = "dueling_dqn",
    **kwargs
):
    config = ExperimentConfig(
        # Training parameters
        episodes=episodes,
        max_steps=max_steps,
        seed=seed,

        # Agent parameters
        batch_size=batch_size,
        memory_size=memory_size,
        lr=lr,

        # Environment parameters
        detect_range=detect_range,

        # Experiment metadata
        name=name,
        **kwargs
    )

    experiment = DuelingDQNExperiment(config, experiment_name="Dueling_DQN_Experiments")
    return experiment.run()
