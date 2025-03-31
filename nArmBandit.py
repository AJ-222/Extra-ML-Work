import numpy as np
from metrics import benchmark, relativePerformance
from Bandit import nArmBandit
from baseAgents import randomAgent, maxGreedAgent

BOLD, RESET = "\033[1m", "\033[0m"

def runAgent(bandit, agent_fn, n_actions=100):
    reward = agent_fn(bandit, n_actions)
    return {
        'reward': reward,
        'benchmark': benchmark(reward),
        'relPerf': relativePerformance(reward, 
                                        bandit.estimatedRewards[bandit.worstBanditArm],
                                        bandit.estimatedRewards[bandit.bestBanditArm]),
    }

def simulate():
    bandit = nArmBandit()
    return {
        'banditData': {
            'estimatedRewards': bandit.estimatedRewards,
            'average': bandit.averageReward,
            'best': bandit.estimatedRewards[bandit.bestBanditArm],
            'worst': bandit.estimatedRewards[bandit.worstBanditArm]
        },
        'random': runAgent(bandit, randomAgent),
        'greedy': runAgent(bandit, maxGreedAgent)
    }

def main():
    simCount = 100
    print(f"{BOLD}Simulating {simCount} times...{RESET}")
    
    results = [simulate() for _ in range(simCount)]
    
    # Calculate averages
    avg = {
        'bandit': {
            'rewardAvg': np.mean([r['banditData']['average'] for r in results]),
            'best': np.mean([r['banditData']['best'] for r in results]),
            'worst': np.mean([r['banditData']['worst'] for r in results])
        },
        'random': {
            k: np.mean([r['random'][k] for r in results])
            for k in ['reward', 'benchmark', 'relPerf']
        },
        'greedy': {
            k: np.mean([r['greedy'][k] for r in results])
            for k in ['reward', 'benchmark', 'relPerf']
        }
    }
    
    # Print results
    print(f"\n\n\n\n\n{BOLD}Average Results Across {simCount} Simulations{RESET}")
    print(f"{'#'*55}")
    print(f"{BOLD}General Information{RESET}")
    print(f"Number of arms: {len(results[0]['banditData']['estimatedRewards'])}")
    print(f"Mean Reward: {avg['bandit']['rewardAvg']:.4f}")
    print(f"Best expected average: {avg['bandit']['best']:.4f}")
    print(f"Worst expected average: {avg['bandit']['worst']:.4f}")
    
    print(f"{'#'*55}\n{BOLD}Basic Agents{RESET}")
    for agent in ['random', 'greedy']:
        print(f"\n{'Random' if agent == 'random' else 'Max Greedy'} Agent:")
        data = avg[agent]
        improvement = data['benchmark'] - avg['random']['benchmark']
        print(f"Average reward: {data['reward']:.4f}")
        print(f"Performance: {data['benchmark']:.2f}%")
        print(f"Improvement vs Random: {improvement:+.2f}%")
        print(f"Performance Relative to Arm Ranges: {data['relPerf']:.2f}%")

if __name__ == "__main__":
    main()