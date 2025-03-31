import numpy as np


class nArmBandit:
    def __init__(self):
        self.arms = [banditArm(i) for i in range(5)]
        self.estimated_rewards = [0] * len(self.arms)
        for i in range(len(self.arms)):
            estimated_reward = 0
            for j in range(100):
                estimated_reward += self.arms[i].getReward()
            estimated_reward /= 100
            self.estimated_rewards[i] = estimated_reward
        self.estimated_rewards = np.round(np.array(self.estimated_rewards),2)
        print("Estimated rewards: ", self.estimated_rewards)
class banditArm:
    def __init__(self, arm_id):
        self.min = np.random.uniform(0, 0.4)
        self.max = np.random.uniform(0.6, 1.0)
        self.mean = np.random.uniform(self.min, self.max)
        self.std = np.random.uniform(0.1, 0.5)
        self.arm_id = arm_id

    def getReward(self):
        return np.random.normal(self.mean, self.std) + np.random.normal(0, 0.1)

print("Bandit Arm Simulation")
print("Number of arms: 5")
print("Estimated rewards:")
nArmBandit()

