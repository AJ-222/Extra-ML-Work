import numpy as np

########################################################
#bandit setup
########################################################
class nArmBandit:
    def __init__(self):
        self.arms = [banditArm(i) for i in range(10)]
        self.estimatedRewards = np.array([arm.mean for arm in self.arms])
        self.estimatedRewards = np.array(self.estimatedRewards)
        self.averageReward = np.mean(self.estimatedRewards)
        self.bestBanditArm = np.argmax(self.estimatedRewards)
        self.worstBanditArm = np.argmin(self.estimatedRewards)

class banditArm:
    def __init__(self, armID):
        if np.random.random() < 0.3:
            if np.random.random() < 0.5:  # Bad arm
                self.min = np.random.uniform(0, 0.2)
                self.max = np.random.uniform(0.2, 0.4)
            else:  # Great arm
                self.min = np.random.uniform(0.7, 0.9)
                self.max = np.random.uniform(0.9, 1.0)
        else:  # Normal arm
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
        self.nArms = len(bandit.arms)
        self.total_reward = 0
        self.action_history = []
        self.reward_history = []

    def selectAction(self):
        action = np.random.randint(0, self.nArms)
        self.action_history.append(action)
        return action

    def update(self, reward):
        self.reward_history.append(reward)
        self.total_reward += reward

def PerformanceMetric(avg, min, max):
    if max == min:
        return 100 if avg >= max else 0
    performance = (avg - min) / (max - min) * 100
    return np.clip(performance, 0, 100)  # Ensure between 0% and 100%



def main():
    bandit = nArmBandit()
    print("Bandit Arm Simulation")
    print("Number of arms: 10")
    print("Estimated rewards:", np.round(bandit.estimatedRewards, 2))
    baseline = Agent(bandit)
    ###########################################
    #randomAgent
    for i in range(10):
        action = baseline.selectAction()
        reward = bandit.arms[action].getReward()
        baseline.total_reward += reward
    baselineReward = baseline.total_reward / 10
    performance = PerformanceMetric(baselineReward, bandit.estimatedRewards[bandit.worstBanditArm], bandit.estimatedRewards[bandit.bestBanditArm])
    ###########################################
    print(f"Random reward: {baselineReward:.2f}")
    print(f"Average reward: {bandit.averageReward:.2f}")
    print(f"Best expected average: {bandit.estimatedRewards[bandit.bestBanditArm]:.2f}")
    print(f"Worst expected average: {bandit.estimatedRewards[bandit.worstBanditArm]:.2f}")
    print(f"Performance metric: {performance:.2f}%")


if __name__ == "__main__":
    main()