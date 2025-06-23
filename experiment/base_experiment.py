import json
import time
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
import torch
from dataclasses import dataclass, asdict
import mlflow


@dataclass
class ExperimentConfig:
    # Training parameters
    episodes: int = 500
    max_steps: int = 100
    seed: int = 42

    # Agent parameters
    batch_size: int = 64
    memory_size: int = 10000
    lr: float = 0.01
    gamma: float = 1.0
    epsilon_init: float = 0.95
    epsilon_min: float = 0.2
    hidden_dim: int = 128
    target_sync_freq: int = 50

    # Environment parameters
    detect_range: int = 4
    goal_threshold: float = 15.0
    use_distance: bool = True
    use_direction: bool = False
    use_stalling: bool = False
    destination: int = 1

    # Experiment parameters
    name: str = ""
    description: str = ""
    tags: List[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        self.validate()

    def validate(self):
        if self.episodes <= 0:
            raise ValueError("Episodes must be positive")
        if self.max_steps <= 0:
            raise ValueError("Max steps must be positive")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if self.memory_size <= 0:
            raise ValueError("Memory size must be positive")
        if not (0 < self.lr <= 1):
            raise ValueError("Learning rate must be in (0, 1]")
        if not (0 <= self.gamma <= 1):
            raise ValueError("Gamma must be in [0, 1]")
        if not (0 <= self.epsilon_min <= self.epsilon_init <= 1):
            raise ValueError("Epsilon values must satisfy 0 <= min <= init <= 1")


@dataclass
class ExperimentMetrics:
    # Performance metrics
    best_reward: float = -float('inf')
    best_episode: int = 0
    avg_reward: float = 0.0
    std_reward: float = 0.0
    final_avg_reward: float = 0.0  # Last 10% of episodes

    # Goal-related metrics
    num_goals: int = 0
    goal_rate: float = 0.0
    avg_steps_to_goal: float = 0.0
    best_steps_to_goal: Optional[int] = None

    # Training metrics
    total_steps: int = 0
    training_time: float = 0.0
    convergence_episode: Optional[int] = None

    # Action distribution
    action_counts: List[int] = None
    action_percentages: List[float] = None

    # Learning progress
    reward_trend: str = "unknown"  # "improving", "stable", "declining"
    learning_stability: float = 0.0  # Coefficient of variation

    def __post_init__(self):
        if self.action_counts is None:
            self.action_counts = []
        if self.action_percentages is None:
            self.action_percentages = []


class BaseExperiment(ABC):
    def __init__(self, config: ExperimentConfig, results_dir: str = "experiment_results",
                 experiment_name: str = "DQN_Experiments"):
        self.config = config
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True, parents=True)

        self._setup_mlflow(experiment_name)

        # Setup logging (keep basic logging for console output)
        self.logger = self._setup_logging()

        # Initialize tracking
        self.metrics = ExperimentMetrics()
        self.rewards_history = []
        self.experiment_id = self._generate_experiment_id()

        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._set_seeds()

        self.logger.info(f"Initialized experiment: {self.experiment_id}")
        self.logger.info(f"Device: {self.device}")

        if mlflow.active_run():
            self.mlflow_run_id = mlflow.active_run().info.run_id
            self.logger.info(f"MLflow run ID: {self.mlflow_run_id}")

            # Log configuration to MLflow
            mlflow.log_params(asdict(self.config))
            mlflow.set_tag("experiment_id", self.experiment_id)
            mlflow.set_tag("device", str(self.device))
        else:
            self.mlflow_run_id = None
            self.logger.info("Using file-based logging (MLflow not available)")

    def _setup_mlflow(self, experiment_name: str):
        """Setup MLflow experiment tracking."""
        # Set tracking URI to base directory (not inside experiment_results)
        mlruns_path = Path("mlruns")
        mlruns_path.mkdir(exist_ok=True, parents=True)
        mlflow.set_tracking_uri(str(mlruns_path.absolute().as_uri()))

        mlflow.set_experiment(experiment_name)

        # Create intuitive run name
        run_name = self._create_run_name()

        # Start MLflow run with descriptive name
        mlflow.start_run(run_name=run_name)

    def _create_run_name(self) -> str:
        name_parts = []

        # Add experiment name if provided
        if self.config.name:
            name_parts.append(self.config.name)

        # Add key hyperparameters
        key_params = [
            ("lr", self.config.lr),
            ("bs", self.config.batch_size),
            ("eps", self.config.episodes),
            ("det", self.config.detect_range)
        ]

        for param_name, value in key_params:
            if value != getattr(ExperimentConfig(), param_name, None):  # Only add if not default
                if isinstance(value, float):
                    name_parts.append(f"{param_name}={value:.4g}")
                else:
                    name_parts.append(f"{param_name}={value}")

        # Add environment features if enabled
        features = []
        if self.config.use_distance:
            features.append("dist")
        if self.config.use_direction:
            features.append("dir")
        if self.config.use_stalling:
            features.append("stall")

        if features:
            name_parts.append(f"feat={'+'.join(features)}")

        # Add destination if not default
        if self.config.destination != 1:
            name_parts.append(f"dest={self.config.destination}")

        # Create final name
        if name_parts:
            base_name = "_".join(name_parts)
        else:
            base_name = "experiment"

        # Add timestamp for uniqueness
        timestamp = datetime.now().strftime('%m%d_%H%M')
        return f"{base_name}_{timestamp}"

    def _generate_experiment_id(self) -> str:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        config_hash = hash(str(asdict(self.config))) % 10000
        name_part = f"_{self.config.name}" if self.config.name else ""
        return f"exp{name_part}_{config_hash}_{timestamp}"

    def _setup_logging(self) -> logging.Logger:
        """Setup experiment-specific logging."""
        logger = logging.getLogger(f"experiment_{id(self)}")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            # File handler
            log_file = self.results_dir / f"experiment_{datetime.now().strftime('%Y%m%d')}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)

            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)

            # Formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)

            logger.addHandler(file_handler)
            logger.addHandler(console_handler)

        return logger

    def _set_seeds(self):
        """Set random seeds for reproducibility."""
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.seed)

    @abstractmethod
    def create_agent(self, state_dim: int, action_dim: int):
        """Create and return the agent for this experiment."""
        pass

    @abstractmethod
    def create_environment(self):
        """Create and return the environment for this experiment."""
        pass

    def run(self) -> Dict[str, Any]:
        """Run the complete experiment."""
        self.logger.info("Starting experiment...")
        start_time = time.time()

        try:
            # Create environment and agent
            env = self.create_environment()
            obs, _ = env.reset(seed=self.config.seed)
            state_dim = obs.shape[0]
            action_dim = env.action_space.n

            agent = self.create_agent(state_dim, action_dim)

            # Run training
            self._train(env, agent)

            # Calculate final metrics
            self.metrics.training_time = time.time() - start_time
            self._calculate_final_metrics()

            # Save results
            results = self._save_results(agent)

            self.logger.info("Experiment completed successfully")
            self._print_summary()

            return results

        except Exception as e:
            self.logger.error(f"Experiment failed: {str(e)}")
            if mlflow.active_run():
                mlflow.log_param("status", "failed")
                mlflow.log_param("error", str(e))
            raise
        finally:
            # End MLflow run (if available)
            if mlflow.active_run():
                mlflow.end_run()

    def _train(self, env, agent):
        self.logger.info("Starting training loop...")

        # Initialize tracking
        action_counts = [0] * env.action_space.n
        best_path = None
        convergence_window = []
        convergence_threshold = 0.05  # 5% improvement threshold

        # Epsilon decay setup
        eps = self.config.epsilon_init
        decay_episodes = int(self.config.episodes * 0.8)
        eps_decay = (self.config.epsilon_init - self.config.epsilon_min) / max(1, decay_episodes)

        for episode in range(self.config.episodes):
            obs, _ = env.reset()
            state = obs
            episode_reward = 0.0
            steps_taken = 0

            for step in range(self.config.max_steps):
                # Action selection and execution
                action = agent.select_action(state, eps)
                action_counts[action] += 1

                next_obs, reward, terminated, truncated, _ = env.step(action)
                next_state = next_obs
                done = terminated or truncated

                episode_reward += reward
                steps_taken += 1
                self.metrics.total_steps += 1

                # Agent learning
                agent.push(state, action, reward, next_state, done)
                agent.update()

                state = next_state

                if terminated:
                    self.metrics.num_goals += 1
                    if self.metrics.best_steps_to_goal is None or steps_taken < self.metrics.best_steps_to_goal:
                        self.metrics.best_steps_to_goal = steps_taken
                        best_path = getattr(env, 'path', None)

                if done:
                    break

            # Epsilon decay
            if episode < decay_episodes:
                eps = max(self.config.epsilon_min, eps - eps_decay)
            else:
                eps = self.config.epsilon_min

            # Record episode results
            self.rewards_history.append(episode_reward)

            # Update best performance
            if episode_reward > self.metrics.best_reward:
                self.metrics.best_reward = episode_reward
                self.metrics.best_episode = episode

            # Check for convergence
            convergence_window.append(episode_reward)
            if len(convergence_window) > 50:
                convergence_window.pop(0)
                if len(convergence_window) == 50 and self.metrics.convergence_episode is None:
                    recent_avg = np.mean(convergence_window[-25:])
                    early_avg = np.mean(convergence_window[:25])
                    if abs(recent_avg - early_avg) / max(abs(early_avg), 1) < convergence_threshold:
                        self.metrics.convergence_episode = episode

            # Progress logging
            if (episode + 1) % 50 == 0:
                recent_avg = np.mean(self.rewards_history[-50:])
                self.logger.info(
                    f"Episode {episode + 1}/{self.config.episodes}, "
                    f"Recent avg reward: {recent_avg:.2f}, "
                    f"Best: {self.metrics.best_reward:.2f}, "
                    f"Goals: {self.metrics.num_goals}, "
                    f"Epsilon: {eps:.3f}"
                )

                mlflow.log_metrics({
                    "episode_recent_avg_reward": recent_avg,
                    "episode_best_reward": self.metrics.best_reward,
                    "episode_num_goals": self.metrics.num_goals,
                    "episode_epsilon": eps,
                    "episode_goal_rate": self.metrics.num_goals / (episode + 1)
                }, step=episode + 1)

            # Log individual episode reward
            if (episode + 1) % 10 == 0 and mlflow.active_run():
                mlflow.log_metric("episode_reward", episode_reward, step=episode + 1)

        # Store action distribution
        self.metrics.action_counts = action_counts
        total_actions = sum(action_counts)
        self.metrics.action_percentages = [
            (count / total_actions * 100) if total_actions > 0 else 0
            for count in action_counts
        ]

    def _calculate_final_metrics(self):
        if not self.rewards_history:
            return

        rewards = np.array(self.rewards_history)

        # Basic statistics
        self.metrics.avg_reward = float(np.mean(rewards))
        self.metrics.std_reward = float(np.std(rewards))

        # Final performance (last 10% of episodes)
        final_episodes = max(1, int(len(rewards) * 0.1))
        self.metrics.final_avg_reward = float(np.mean(rewards[-final_episodes:]))

        # Goal metrics
        if self.config.episodes > 0:
            self.metrics.goal_rate = self.metrics.num_goals / self.config.episodes

        if self.metrics.num_goals > 0:
            self.metrics.avg_steps_to_goal = self.metrics.total_steps / self.metrics.num_goals

        # Learning stability
        if len(rewards) > 1:
            self.metrics.learning_stability = float(np.std(rewards) / max(abs(np.mean(rewards)), 1))

        # Reward trend analysis
        if len(rewards) >= 100:
            early_avg = np.mean(rewards[:50])
            late_avg = np.mean(rewards[-50:])
            improvement = (late_avg - early_avg) / max(abs(early_avg), 1)

            if improvement > 0.1:
                self.metrics.reward_trend = "improving"
            elif improvement < -0.1:
                self.metrics.reward_trend = "declining"
            else:
                self.metrics.reward_trend = "stable"

    def _save_results(self, agent) -> Dict[str, Any]:
        model_path = self.results_dir / f"model_{self.experiment_id}.pth"
        agent.save(str(model_path))

        try:
            mlflow.pytorch.log_model(agent.q_network, "model")
        except Exception as e:
            self.logger.warning(f"Could not log model to MLflow: {e}")

        final_metrics = asdict(self.metrics)
        for key, value in final_metrics.items():
            if isinstance(value, (int, float)):
                try:
                    mlflow.log_metric(f"final_{key}", value)
                except Exception as e:
                    self.logger.warning(f"Could not log metric {key}: {e}")

        # Log rewards history as artifact
        rewards_path = self.results_dir / f"rewards_{self.experiment_id}.json"
        with open(rewards_path, 'w') as f:
            json.dump(self.rewards_history, f)

        try:
            mlflow.log_artifact(str(rewards_path))
        except Exception as e:
            self.logger.warning(f"Could not log rewards artifact: {e}")

        # Create results dictionary
        results = {
            'experiment_id': self.experiment_id,
            'mlflow_run_id': self.mlflow_run_id,
            'config': asdict(self.config),
            'metrics': asdict(self.metrics),
            'rewards_history': self.rewards_history,
            'model_path': str(model_path),
            'timestamp': datetime.now().isoformat(),
            'device': str(self.device)
        }

        # Save results locally with custom JSON encoder for infinity handling
        results_path = self.results_dir / f"results_{self.experiment_id}.json"

        def json_serializer(obj):
            if isinstance(obj, float) and (obj == float('inf') or obj == float('-inf')):
                return None
            return str(obj)

        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=json_serializer)

        try:
            mlflow.log_artifact(str(results_path))
            self.logger.info(f"Results logged to MLflow run: {self.mlflow_run_id}")
        except Exception as e:
            self.logger.warning(f"Could not log results artifact: {e}")

        self.logger.info(f"Results saved locally to: {results_path}")

        return results

    def _print_summary(self):
        print(f"\n{'='*80}")
        print(f"EXPERIMENT SUMMARY: {self.experiment_id}")
        print(f"{'='*80}")
        print(f"Configuration:")
        print(f"  Episodes: {self.config.episodes}")
        print(f"  Max Steps: {self.config.max_steps}")
        print(f"  Learning Rate: {self.config.lr}")
        print(f"  Batch Size: {self.config.batch_size}")
        print(f"  Memory Size: {self.config.memory_size}")
        print(f"\nPerformance Metrics:")
        print(f"  Best Reward: {self.metrics.best_reward:.2f} (Episode {self.metrics.best_episode})")
        print(f"  Average Reward: {self.metrics.avg_reward:.2f} ± {self.metrics.std_reward:.2f}")
        print(f"  Final Average: {self.metrics.final_avg_reward:.2f}")
        print(f"  Learning Trend: {self.metrics.reward_trend}")
        print(f"\nGoal Achievement:")
        print(f"  Goals Reached: {self.metrics.num_goals}/{self.config.episodes} ({self.metrics.goal_rate:.1%})")
        if self.metrics.num_goals > 0:
            print(f"  Avg Steps to Goal: {self.metrics.avg_steps_to_goal:.1f}")
            print(f"  Best Steps to Goal: {self.metrics.best_steps_to_goal}")
        print(f"\nTraining Info:")
        print(f"  Total Steps: {self.metrics.total_steps}")
        print(f"  Training Time: {self.metrics.training_time:.1f}s")
        if self.metrics.convergence_episode:
            print(f"  Convergence Episode: {self.metrics.convergence_episode}")
        print(f"\nAction Distribution:")
        for i, (count, pct) in enumerate(zip(self.metrics.action_counts, self.metrics.action_percentages)):
            print(f"  Action {i}: {count} times ({pct:.1f}%)")
        print(f"{'='*80}")
