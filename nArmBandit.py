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
        if np.random.random() < 0.3:    #rare cases of really bad or good arms
            if np.random.random() < 0.5:  #bad
                self.min = np.random.uniform(0, 0.19)
                self.max = np.random.uniform(0.21, 0.4)
            else:  #good
                self.min = np.random.uniform(0.7, 0.89)
                self.max = np.random.uniform(0.9, 1.0)
        else:  #standard
            self.min = np.random.uniform(0, 0.6)
            self.max = np.random.uniform(self.min + 0.1, 1.0)
        self.mean = np.random.uniform(self.min, self.max)
        
        self.std = np.random.uniform(0.1, 0.5)
        self.armID = armID



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

def performanceMetric(avg, min, max):
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
        
def main():
    bandit = nArmBandit()
    ############################################
    #basic Agents
    ###########################################
    baseline = randomAgent(bandit, 100)
    randPerf = performanceMetric(baseline, bandit.estimatedRewards[bandit.worstBanditArm], bandit.estimatedRewards[bandit.bestBanditArm])
    randToAvg = performanceMetric(baseline, bandit.averageReward, bandit.estimatedRewards[bandit.bestBanditArm])
    maxGreedy = maxGreedAgent(bandit, 100)
    maxPerf = performanceMetric(maxGreedy, bandit.estimatedRewards[bandit.worstBanditArm], bandit.estimatedRewards[bandit.bestBanditArm])
    maxToAvg = performanceMetric(maxGreedy, bandit.averageReward, bandit.estimatedRewards[bandit.bestBanditArm])
    ###########################################
    print(f"{BOLD}Bandit Arm Simulation{RESET}")
    print(f"Number of arms: 10")
    print(f"#####################################################")
    print(f"{BOLD}General Information{RESET}")
    print(f"Estimated rewards: {np.round(bandit.estimatedRewards, 2)}")
    print(f"Average reward: {bandit.averageReward:.2f}")
    print(f"Best expected average: {bandit.estimatedRewards[bandit.bestBanditArm]:.2f}")
    print(f"Worst expected average: {bandit.estimatedRewards[bandit.worstBanditArm]:.2f}")
    print(f"#####################################################\n{BOLD}Basic Agents{RESET}")
    print(f"Random Agents (pulls arms randomly):")
    print(f"Random reward: {baseline:.2f}")
    print(f"Performance: {randPerf:.2f}%")
    if randToAvg >= 0:
        print(f"Comparison to Average: +{randToAvg:.2f}%")
    else:
        print(f"Comparison to Average: {randToAvg:.2f}%")
    print(f"Max Greedy Agent (checks all arms once then pulls the best arm based on 1 loop of exploration):")
    print(f"Max greedy reward: {maxGreedy:.2f}")
    print(f"Performance: {maxPerf:.2f}%")
    if maxToAvg >= 0:
        print(f"Comparison to Average: +{maxToAvg:.2f}%")
    else:
        print(f"Comparison to Average: {maxToAvg:.2f}%")
    




if __name__ == "__main__":
    main()