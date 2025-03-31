import numpy as np

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