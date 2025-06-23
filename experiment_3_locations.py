from experiment import run_dueling_dqn_experiment, run_dqn_experiment
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser(description="Location Comparison Experiments")
    parser.add_argument("--agent_type", type=str, choices=["dqn", "dueling_dqn"],
                        required=True, help="Type of DQN agent to use")
    return parser.parse_args()


def run_location_experiments(agent_type):
    base_config = {
        'episodes': 500,
        'max_steps': 200,
        'batch_size': 32,
        'memory_size': 10000,
        'lr': 0.0005,
        'detect_range': 4,
        'use_distance': True,
        'use_direction': False,
        'use_stalling': False,
        'seed': 42
    }

    locations = [
        {
            'destination': 0,
            'name': 'EASY_LOCATION',
            'description': 'Easy nearby location for testing'
        },
        {
            'destination': 1,
            'name': 'Markthal',
            'description': 'Markthal'
        },
        {
            'destination': 2,
            'name': 'Auditorium',
            'description': 'Auditorium building'
        },
        {
            'destination': 3,
            'name': 'Nexus',
            'description': 'Nexus building'
        }
    ]

    total_experiments = len(locations)

    print(f"Starting Experiment 3: Location Comparison")
    print(f"Agent Type: {agent_type.upper()}")
    print(f"Total experiments to run: {total_experiments}")
    print(f"Episodes: {base_config['episodes']}, Max Steps: {base_config['max_steps']}")
    print(f"Using default hyperparameters and distance rewards")
    print("=" * 80)

    results = []

    for i, location in enumerate(locations, 1):
        print(f"\nRunning experiment {i}/{total_experiments}")
        print(f"Location: {location['description']} (destination={location['destination']})")

        try:
            # Create experiment config for this location
            experiment_config = {
                **base_config,
                'name': f"location_{location['name'].lower()}",
                'destination': location['destination']
            }

            experiment_func = run_dueling_dqn_experiment if agent_type == "dueling_dqn" else run_dqn_experiment

            # Run experiment
            result = experiment_func(**experiment_config)

            # Store results
            experiment_summary = {
                'experiment_id': result['experiment_id'],
                'destination': location['destination'],
                'location_name': location['name'],
                'description': location['description'],
                'best_reward': result['metrics']['best_reward'],
                'goal_rate': result['metrics']['goal_rate'],
                'avg_reward': result['metrics']['avg_reward'],
                'final_avg_reward': result['metrics']['final_avg_reward'],
                'best_steps_to_goal': result['metrics']['best_steps_to_goal'],
                'avg_steps_to_goal': result['metrics']['avg_steps_to_goal'],
                'training_time': result['metrics']['training_time'],
                'convergence_episode': result['metrics']['convergence_episode']
            }
            results.append(experiment_summary)

            goal_info = ""
            if result['metrics']['best_steps_to_goal'] is not None:
                goal_info = f", Best steps to goal: {result['metrics']['best_steps_to_goal']}"

            print(
                f"Completed: Best reward = {result['metrics']['best_reward']:.2f}, Goal rate = {result['metrics']['goal_rate']:.1%}{goal_info}")

        except Exception as e:
            print(f"Failed: {str(e)}")
            continue

    # Print summary
    print("\n" + "=" * 80)
    print("EXPERIMENT 3 SUMMARY")
    print("=" * 80)

    if results:
        print(f"Completed {len(results)}/{total_experiments} experiments successfully")
        print("\nResults by location:")

        # Sort by goal rate first, then by best reward
        results.sort(key=lambda x: (x['goal_rate'], x['best_reward']), reverse=True)

        for i, result in enumerate(results, 1):
            print(f"{i}. {result['description']} (dest={result['destination']})")
            print(f"   Best reward: {result['best_reward']:.2f}")
            print(f"   Goal rate: {result['goal_rate']:.1%}")
            print(f"   Avg reward: {result['avg_reward']:.2f}")
            print(f"   Final avg: {result['final_avg_reward']:.2f}")

            if result['best_steps_to_goal'] is not None:
                print(f"   Best steps to goal: {result['best_steps_to_goal']}")
            if result['avg_steps_to_goal'] > 0:
                print(f"   Avg steps to goal: {result['avg_steps_to_goal']:.1f}")
            if result['convergence_episode'] is not None:
                print(f"   Converged at episode: {result['convergence_episode']}")

            print(f"   Training time: {result['training_time']:.1f}s")
            print()

        # Analysis by difficulty
        print("Location Difficulty Analysis:")
        print("=" * 40)

        # Sort by goal rate to see difficulty
        by_success = sorted(results, key=lambda x: x['goal_rate'], reverse=True)

        if by_success[0]['goal_rate'] > 0:
            print(f"Easiest: {by_success[0]['description']} (goal rate: {by_success[0]['goal_rate']:.1%})")
        if by_success[-1]['goal_rate'] >= 0:
            print(f"Hardest: {by_success[-1]['description']} (goal rate: {by_success[-1]['goal_rate']:.1%})")

        # Sort by reward to see learning performance
        by_reward = sorted(results, key=lambda x: x['best_reward'], reverse=True)
        print(f"Best learning: {by_reward[0]['description']} (best reward: {by_reward[0]['best_reward']:.2f})")

        # Convergence analysis
        converged = [r for r in results if r['convergence_episode'] is not None]
        if converged:
            fastest_convergence = min(converged, key=lambda x: x['convergence_episode'])
            print(
                f"Fastest convergence: {fastest_convergence['description']} (episode {fastest_convergence['convergence_episode']})")

    else:
        print("No experiments completed successfully!")

    return results


if __name__ == "__main__":
    args = parse_args()
    run_location_experiments(agent_type=args.agent_type)
