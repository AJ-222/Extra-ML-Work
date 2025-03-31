import numpy as np

########################################################
#bandit setup
########################################################
class nArmBandit:
    def __init__(self):
        self.arms = [banditArm(i) for i in range(10)]
        self.estimatedRewards = [0] * len(self.arms)
        for i in range(len(self.arms)):
            self.estimatedRewards[i] = np.mean([self.arms[i].getReward() for _ in range(1000)])
        self.estimatedRewards = np.array(self.estimatedRewards)
        self.averageReward = np.mean(self.estimatedRewards)
        self.bestBanditArm = np.argmax(self.estimatedRewards)
        self.worstBanditArm = np.argmin(self.estimatedRewards)

class banditArm:
    def __init__(self, armID):
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
    def __init__(self, bandit, action):
        self.bandit = bandit
        self.nArms = len(bandit.arms)
        self.action = action
        self.reward = 0
        self.sum = 0
        self.actionHistory = []
        self.rewardHistory = []


    def selectAction(self):
        self.action = np.random.randint(0, self.nArms)
        return self.action

def PerformanceMetric(avg,min,max):
    return (avg - min) / (max - min)



def main():
    bandit = nArmBandit()
    print("Bandit Arm Simulation")
    print("Number of arms: 5")
    print("Estimated rewards:", np.round(bandit.estimatedRewards, 2))
    baseline = Agent(bandit, np.random.randint(0, 5))
    ###########################################
    #randomAgent
    for i in range(10):
        action = baseline.selectAction()
        reward = bandit.arms[action].getReward()
        baseline.sum += reward
    baselineReward = baseline.sum / 10
    performance = PerformanceMetric(baselineReward, bandit.estimatedRewards[bandit.worstBanditArm], bandit.estimatedRewards[bandit.bestBanditArm]) * 100
    ###########################################
    print(f"Random reward: {baselineReward:.2f}")
    print(f"Average reward: {bandit.averageReward:.2f}")
    print(f"Best expected average: {bandit.estimatedRewards[bandit.bestBanditArm]:.2f}")
    print(f"Worst expected average: {bandit.estimatedRewards[bandit.worstBanditArm]:.2f}")
    print(f"Performance metric: {performance:.2f}%")
if __name__ == "__main__":
    main()