import numpy as np

BOLD = "\033[1m"
RESET = "\033[0m"
########################################################
#bandit setup
########################################################
class nArmBandit:
    def __init__(self):
        self.arms = [banditArm(i) for i in range(10)]
        self.nArms = len(self.arms)
        self.estimatedRewards = np.array([arm.mean for arm in self.arms])
        self.averageReward = np.mean(self.estimatedRewards)
        self.bestBanditArm = np.argmax(self.estimatedRewards)
        self.worstBanditArm = np.argmin(self.estimatedRewards)

class banditArm:
    def __init__(self, armID):
        quality = np.random.random()
        if quality < 0.15:  # 15% bad arms
            self.min, self.max = 0, 0.3
        elif quality > 0.85:  # 15% good arms
            self.min, self.max = 0.7, 1.0
        else:  # Mostly normal arms
            self.min, self.max = 0.3, 0.7
        self.mean = np.random.uniform(self.min*1.1, self.max*0.9)
        self.std = np.random.uniform(0.2) #fixed for development



    def getReward(self):
        reward = np.random.normal(self.mean, self.std)
        return np.clip(reward, self.min, self.max)

#######################################################
#basic agents
#######################################################
class Agent:
    def __init__(self, bandit):
        self.bandit = bandit
        self.totalReward = 0
        self.action_history = []
        self.reward_history = []

    def update(self, action, reward):
        self.action_history.append(action)
        self.reward_history.append(reward)
        self.totalReward += reward

MIN_BENCHMARK = 0
MAX_BENCHMARK = 1
def benchmark(avg):
    if MAX_BENCHMARK == MIN_BENCHMARK:
        return 100 if avg >= MAX_BENCHMARK else 0
    performance = (avg - MIN_BENCHMARK) / (MAX_BENCHMARK - MIN_BENCHMARK) * 100
    return np.clip(performance, 0, 100)

def relativePerformance(avg, min, max):
    if max == min:
        return 100 if avg >= max else 0
    performance = (avg - min) / (max - min) * 100
    return np.clip(performance, 0, 100)

def randomAgent(bandit, nActions):
    baselineAgent = Agent(bandit)
    for _ in range(nActions):
        action = np.random.randint(0, bandit.nArms)
        baselineAgent.action_history.append(action)
        reward = bandit.arms[action].getReward()
        baselineAgent.update(action, reward)
    baselineAgent.avgReward = baselineAgent.totalReward / nActions
    return baselineAgent.avgReward

def maxGreedAgent(bandit, nActions):
    if nActions < bandit.nArms:
        print(f"{BOLD}Warning: {RESET}Number of arms exceeds number of actions. Using only {bandit.nArms} actions.")
        return 0
    greedyAgent = Agent(bandit)
    bestArm = [0,0]
    for x in range(bandit.nArms):
        action = x
        reward = bandit.arms[x].getReward()
        greedyAgent.update(x,reward)
        if reward > bestArm[1]:
            bestArm[0] = x
            bestArm[1] = reward
    for _ in range(nActions - bandit.nArms):
        action = bestArm[0]
        reward = bandit.arms[action].getReward()
        greedyAgent.update(action, reward)
    greedyAgent.avgReward = greedyAgent.totalReward / nActions
    return greedyAgent.avgReward
        
def simulate():
    bandit = nArmBandit()
    #basic Agents
    baseline = randomAgent(bandit, 100)
    randBenchmark = benchmark(baseline)
    randRelPerf = relativePerformance(baseline, bandit.estimatedRewards[bandit.worstBanditArm], bandit.estimatedRewards[bandit.bestBanditArm])
    randToAvg = relativePerformance(baseline, bandit.averageReward, bandit.estimatedRewards[bandit.bestBanditArm])
    maxGreedy = maxGreedAgent(bandit, 100)
    maxGreedyBenchmark = benchmark(maxGreedy)
    maxRelPerf = relativePerformance(maxGreedy, bandit.estimatedRewards[bandit.worstBanditArm], bandit.estimatedRewards[bandit.bestBanditArm])
    maxToAvg = relativePerformance(maxGreedy, bandit.averageReward, bandit.estimatedRewards[bandit.bestBanditArm])
    
    return {
        'estimatedRewards': bandit.estimatedRewards,
        'averageReward': bandit.averageReward,
        'bestReward': bandit.estimatedRewards[bandit.bestBanditArm],
        'worstReward': bandit.estimatedRewards[bandit.worstBanditArm],
        'randomReward': baseline,
        'randomBenchmark': randBenchmark,
        'randomPerformance': randRelPerf,
        'randToAvg': randToAvg,
        'greedyReward': maxGreedy,
        'greedyBenchmark': maxGreedyBenchmark,
        'greedyPerformance': maxRelPerf,
        'greedyToAverage': maxToAvg
    }

def main():
    num_simulations = 100
    results = []
    print(f"{BOLD}Simulating {num_simulations} times...{RESET}")
    for _ in range(num_simulations):
        results.append(simulate())

    avg_results = {
        'averageReward': np.mean([r['averageReward'] for r in results]),
        'bestReward': np.mean([r['bestReward'] for r in results]),
        'worstReward': np.mean([r['worstReward'] for r in results]),
        'randomReward': np.mean([r['randomReward'] for r in results]),
        'randomBenchmark': np.mean([r['randomBenchmark'] for r in results]),
        'randomPerformance': np.mean([r['randomPerformance'] for r in results]),
        'randToAvg': np.mean([r['randToAvg'] for r in results]),
        'greedyReward': np.mean([r['greedyReward'] for r in results]),
        'greedyBenchmark': np.mean([r['greedyBenchmark'] for r in results]),
        'greedyPerformance': np.mean([r['greedyPerformance'] for r in results]),
        'greedyToAverage': np.mean([r['greedyToAverage'] for r in results]),
    }
    
    #print results
    print(f"\n{BOLD}Average Results Across {num_simulations} Simulations{RESET}")
    print(f"#####################################################")
    print(f"{BOLD}General Information{RESET}")
    print(f"Mean Reward: {avg_results['averageReward']:.2f}")
    print(f"Best expected average: {avg_results['bestReward']:.2f}")
    print(f"Worst expected average: {avg_results['worstReward']:.2f}")
    print(f"#####################################################\n{BOLD}Basic Agents{RESET}")
    
    print(f"Random Agent (pulls arms randomly):")
    print(f"Average reward: {avg_results['randomReward']:.2f}")
    print(f"Performance: {avg_results['randomBenchmark']:.2f}%")
    print(f"Performance Relative To Given Arms: {avg_results['randomPerformance']:.2f}%")
    print(f"Comparison to Average: {avg_results['randToAvg']:+.2f}%")
    
    print(f"\nMax Greedy Agent:")
    print(f"Average reward: {avg_results['greedyReward']:.2f}")
    print(f"Performance: {avg_results['greedyBenchmark']:.2f}%")
    print(f"Performance Relative To Given Arms: {avg_results['greedyPerformance']:.2f}%")
    print(f"Comparison to Average: {avg_results['greedyToAverage']:+.2f}%")

if __name__ == "__main__":
    main()