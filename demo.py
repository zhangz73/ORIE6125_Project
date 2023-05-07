from utils.linear_threshold import LinearThreshold

## Generate a single trial of contagion in the network
linear_threshold = LinearThreshold()
infected, N, z = linear_threshold.trial(n = 1000, all = False)
print(infected, N, z)

## Generate multiple trials of contagion in the network using parallelization
linear_threshold = LinearThreshold()
results = linear_threshold.multi_trials(n = 1000, all = False, num_trials = 10, n_cpu = 2)
print(results)

## Animation
linear_threshold = LinearThreshold()
linear_threshold.animate(thres_up = 0.5, n = 100)
