from Agent import Agent
import numpy as np
BOLD = "\033[1m"
RESET = "\033[0m"
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