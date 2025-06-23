from experiment import run_dueling_dqn_experiment, run_dqn_experiment
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser(description="Reward Type Comparison Experiments")
    parser.add_argument("--agent_type", type=str, choices=["dqn", "dueling_dqn"],
                        required=True, help="Type of DQN agent to use")
    return parser.parse_args()


def run_reward_experiments(agent_type):
    base_config = {
        'episodes': 500,
        'max_steps': 200,
        'batch_size': 32,
        'memory_size': 10000,
        'lr': 0.0005,
        'detect_range': 4,
        'destination': 2,
        'seed': 42
    }

    reward_configs = [
        {
            'name': 'no_extra_rewards',
            'description': 'No extra rewards (baseline)',
            'use_distance': False,
            'use_direction': False,
            'use_stalling': False
        },
        {
            'name': 'distance_rewards',
            'description': 'Distance-based rewards only',
            'use_distance': True,
            'use_direction': False,
            'use_stalling': False
        },
        {
            'name': 'direction_rewards',
            'description': 'Direction-based rewards only',
            'use_distance': False,
            'use_direction': True,
            'use_stalling': False
        },
        {
            'name': 'stalling_rewards',
            'description': 'Stalling penalty only',
            'use_distance': False,
            'use_direction': False,
            'use_stalling': True
        },
        {
            'name': 'distance_direction',
            'description': 'Distance + Direction rewards',
            'use_distance': True,
            'use_direction': True,
            'use_stalling': False
        },
        {
            'name': 'distance_stalling',
            'description': 'Distance + Stalling penalty',
            'use_distance': True,
            'use_direction': False,
            'use_stalling': True
        },
        {
            'name': 'direction_stalling',
            'description': 'Direction + Stalling penalty',
            'use_distance': False,
            'use_direction': True,
            'use_stalling': True
        },
        {
            'name': 'all_rewards',
            'description': 'All reward types enabled',
            'use_distance': True,
            'use_direction': True,
            'use_stalling': True
        }
    ]

    total_experiments = len(reward_configs)

    print(f"Starting Experiment 2: Reward Type Comparison")
    print(f"Agent Type: {agent_type.upper()}")
    print(f"Total experiments to run: {total_experiments}")
    print(f"Location: Auditorium (destination=3)")
    print(f"Episodes: {base_config['episodes']}, Max Steps: {base_config['max_steps']}")
    print(f"Using default hyperparameters from hyperparams.yml")
    print("=" * 80)

    results = []

    for i, config in enumerate(reward_configs, 1):
        print(f"\nRunning experiment {i}/{total_experiments}")
        print(f"Reward type: {config['description']}")
        print(
            f"Features: distance={config['use_distance']}, direction={config['use_direction']}, stalling={config['use_stalling']}")

        try:
            # Merge base config with reward config
            experiment_config = {
                **base_config,
                'name': f"reward_{config['name']}",
                'use_distance': config['use_distance'],
                'use_direction': config['use_direction'],
                'use_stalling': config['use_stalling']
            }

            experiment_func = run_dueling_dqn_experiment if agent_type == "dueling_dqn" else run_dqn_experiment

            result = experiment_func(**experiment_config)

            # Store results
            experiment_summary = {
                'experiment_id': result['experiment_id'],
                'reward_type': config['name'],
                'description': config['description'],
                'use_distance': config['use_distance'],
                'use_direction': config['use_direction'],
                'use_stalling': config['use_stalling'],
                'best_reward': result['metrics']['best_reward'],
                'goal_rate': result['metrics']['goal_rate'],
                'avg_reward': result['metrics']['avg_reward'],
                'final_avg_reward': result['metrics']['final_avg_reward'],
                'training_time': result['metrics']['training_time']
            }
            results.append(experiment_summary)

            print(
                f"Completed: Best reward = {result['metrics']['best_reward']:.2f}, Goal rate = {result['metrics']['goal_rate']:.1%}")

        except Exception as e:
            print(f"Failed: {str(e)}")
            continue

    # Print summary
    print("\n" + "=" * 80)
    print("EXPERIMENT 2 SUMMARY")
    print("=" * 80)

    if results:
        # Sort by best reward
        results.sort(key=lambda x: x['best_reward'], reverse=True)

        print(f"Completed {len(results)}/{total_experiments} experiments successfully")
        print("\nResults by best reward:")
        for i, result in enumerate(results, 1):
            features = []
            if result['use_distance']:
                features.append('dist')
            if result['use_direction']:
                features.append('dir')
            if result['use_stalling']:
                features.append('stall')
            feature_str = '+'.join(features) if features else 'none'

            print(f"{i}. {result['description']}")
            print(f"   Features: {feature_str}")
            print(f"   Best reward: {result['best_reward']:.2f}")
            print(f"   Goal rate: {result['goal_rate']:.1%}")
            print(f"   Avg reward: {result['avg_reward']:.2f}")
            print(f"   Final avg: {result['final_avg_reward']:.2f}")
            print(f"   Training time: {result['training_time']:.1f}s")
            print()

        # Best reward type analysis
        best_result = results[0]
        print(f"Best reward configuration: {best_result['description']}")
        print(f"  Distance rewards: {best_result['use_distance']}")
        print(f"  Direction rewards: {best_result['use_direction']}")
        print(f"  Stalling penalty: {best_result['use_stalling']}")
        print(f"  Best reward: {best_result['best_reward']:.2f}")
        print(f"  Goal rate: {best_result['goal_rate']:.1%}")

    else:
        print("No experiments completed successfully!")

    return results


if __name__ == "__main__":
    args = parse_args()
    run_reward_experiments(agent_type=args.agent_type)
