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