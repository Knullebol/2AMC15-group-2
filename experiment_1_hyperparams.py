from experiment import run_dueling_dqn_experiment, run_dqn_experiment
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser(description="Hyperparameter Tuning Experiments")
    parser.add_argument("--agent_type", type=str, choices=["dqn", "dueling_dqn"],
                        required=True, help="Type of DQN agent to use")
    return parser.parse_args()


def run_hyperparameter_experiments(agent_type):
    defaults = {
        'batch_size': 32,
        'memory_size': 10000,
        'lr': 0.0005,
        'detect_range': 4
    }

    parameter_studies = [
        ('batch_size', [16, 32, 64]),
        ('memory_size', [5000, 10000, 20000]),
        ('lr', [0.0005, 0.001, 0.01]),
        ('detect_range', [4, 8, 10])
    ]

    # Fixed parameters for all experiments
    base_config = {
        'episodes': 500,
        'max_steps': 200,
        'destination': 2,  # Auditorium
        'use_distance': True,  # Use default distance rewards
        'seed': 42
    }

    # Calculate total experiments (sum of all parameter ranges)
    total_experiments = sum(len(values) for _, values in parameter_studies)

    print(f"Starting Experiment 1: Hyperparameter Tuning")
    print(f"Agent Type: {agent_type.upper()}")
    print(f"Total experiments to run: {total_experiments}")
    print(f"Location: Auditorium (destination=3)")
    print(f"Episodes: {base_config['episodes']}, Max Steps: {base_config['max_steps']}")
    print(f"Testing each parameter individually while keeping others at defaults")
    print("=" * 80)

    results = []
    experiment_count = 0

    # Test each parameter individually
    for param_name, param_values in parameter_studies:
        print(f"\n\nTesting {param_name.upper()} parameter:")
        print(f"Values to test: {param_values}")
        print(f"Other parameters fixed at defaults: {defaults}")
        print("-" * 60)

        for value in param_values:
            experiment_count += 1
            print(f"\nRunning experiment {experiment_count}/{total_experiments}")

            # Create config with this parameter value and defaults for others
            experiment_config = defaults.copy()
            experiment_config[param_name] = value

            print(f"Testing {param_name}={value}")
            print(f"Full config: {experiment_config}")

            try:
                # Create experiment name
                name = f"hyperparam_{param_name}_{value}"

                experiment_func = run_dueling_dqn_experiment if agent_type == "dueling_dqn" else run_dqn_experiment

                # Run experiment
                result = experiment_func(
                    name=name,
                    batch_size=experiment_config['batch_size'],
                    memory_size=experiment_config['memory_size'],
                    lr=experiment_config['lr'],
                    detect_range=experiment_config['detect_range'],
                    **base_config
                )

                experiment_summary = {
                    'experiment_id': result['experiment_id'],
                    'parameter_tested': param_name,
                    'parameter_value': value,
                    'batch_size': experiment_config['batch_size'],
                    'memory_size': experiment_config['memory_size'],
                    'lr': experiment_config['lr'],
                    'detect_range': experiment_config['detect_range'],
                    'best_reward': result['metrics']['best_reward'],
                    'goal_rate': result['metrics']['goal_rate'],
                    'avg_reward': result['metrics']['avg_reward'],
                    'final_avg_reward': result['metrics']['final_avg_reward']
                }
                results.append(experiment_summary)

                print(
                    f"Completed: Best reward = {result['metrics']['best_reward']:.2f}, Goal rate = {result['metrics']['goal_rate']:.1%}")

            except Exception as e:
                print(f"Failed: {str(e)}")
                continue

    # Print summary
    print("\n" + "=" * 80)
    print("EXPERIMENT 1 SUMMARY")
    print("=" * 80)

    if results:
        print(f"Completed {len(results)}/{total_experiments} experiments successfully")

        # Analyze results by parameter
        for param_name, param_values in parameter_studies:
            print(f"\n{param_name.upper()} Analysis:")
            print("-" * 40)

            # Get results for this parameter
            param_results = [r for r in results if r['parameter_tested'] == param_name]
            if not param_results:
                continue

            # Sort by best reward for this parameter
            param_results.sort(key=lambda x: x['best_reward'], reverse=True)

            print(f"Tested values: {param_values}")
            print("Results (sorted by best reward):")

            for result in param_results:
                print(f"  {param_name}={result['parameter_value']:>6}: "
                      f"Best reward={result['best_reward']:>6.2f}, "
                      f"Goal rate={result['goal_rate']:>5.1%}, "
                      f"Avg reward={result['avg_reward']:>6.2f}")

            # Find best value for this parameter
            best_param_result = param_results[0]
            print(f"Best {param_name}: {best_param_result['parameter_value']} "
                  f"(reward: {best_param_result['best_reward']:.2f})")

        # Overall best configuration
        print(f"\n{'='*60}")
        print("OVERALL ANALYSIS")
        print(f"{'='*60}")

        # Sort all results by best reward
        results.sort(key=lambda x: x['best_reward'], reverse=True)

        print("\nTop 5 experiments by best reward:")
        for i, result in enumerate(results[:5], 1):
            print(f"{i}. {result['parameter_tested']}={result['parameter_value']}: "
                  f"Best reward={result['best_reward']:.2f}, Goal rate={result['goal_rate']:.1%}")

        # Best value for each parameter
        print(f"\nRecommended parameter values (based on best individual performance):")
        for param_name, _ in parameter_studies:
            param_results = [r for r in results if r['parameter_tested'] == param_name]
            if param_results:
                best_param = max(param_results, key=lambda x: x['best_reward'])
                print(f"  {param_name}: {best_param['parameter_value']} "
                      f"(achieved {best_param['best_reward']:.2f} reward)")

    else:
        print("No experiments completed successfully!")

    return results


if __name__ == "__main__":
    args = parse_args()
    run_hyperparameter_experiments(agent_type=args.agent_type)
