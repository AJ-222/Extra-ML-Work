import numpy as np

MIN_BENCHMARK = 0
MAX_BENCHMARK = 1

def benchmark(avg):
    """Absolute performance metric (0-100 scale)"""
    if MAX_BENCHMARK == MIN_BENCHMARK:
        return 100 if avg >= MAX_BENCHMARK else 0
    performance = (avg - MIN_BENCHMARK) / (MAX_BENCHMARK - MIN_BENCHMARK) * 100
    return np.clip(performance, 0, 100)

def relativePerformance(avg, min, max):
    """Relative performance within a specific bandit instance"""
    if max == min:
        return 100 if avg >= max else 0
    performance = (avg - min) / (max - min) * 100
    return np.clip(performance, 0, 100)

def compareToAverage(avg, bandit_avg, bandit_max):
    """How much better/worse than the bandit's average"""
    if bandit_max == bandit_avg:
        return 0
    return ((avg - bandit_avg) / (bandit_max - bandit_avg)) * 100