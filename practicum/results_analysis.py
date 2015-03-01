# General code for analysis of the validation results

import json
import pandas as pd
import matplotlib as plt

RESULTS_FILE = "/Users/v685573/OneDrive/Documents/MSIA5243/practicum/results.json"
PLOT_STYLES  = [".", "o", "v", "^", ">", "<", "*", "h", "+", "x", "D"]
NUM_TRAIN_RECORDS = [100000, 200000, 400000, 600000, 800000, 1000000, 1300000, 1600000, 2500000, 3200000, 5000000]


with open(RESULTS_FILE, 'r') as f:
    results = json.load(f)

# PLot data ontop of itself
for i in range(len(results)):
    plt.plot(results[str(NUM_TRAIN_RECORDS[i])]['locs'], results[str(NUM_TRAIN_RECORDS[i])]['scores'], PLOT_STYLES[i])


# top scores
x = list()
y_top = list()
y_top5 = list()
for k, v in results.iteritems():
    x.append(k)
    y_top.append(v['top'])
    y_top5.append(v['top5'])
plt.plot(x, y_top, '-x')
plt.plot(x, y_top5, '-D')


# plot separate plots. scatter
f, plots = plt.subplots(3, 4)
for p in range(len(PLOT_STYLES)):
    y = p%4
    x = p/4
    print x, y
    plots[x][y].plot(results['100000']['locs'], results['100000']['scores'], PLOT_STYLES[p])
    plots[x][y].set_title(PLOT_STYLES[p])


# Plot locs
f, plots = plt.subplots(3, 4)
for p in range(len(PLOT_STYLES)):
    y = p%4
    x = p/4
    print x, y
    plots[x][y].hist(results['100000']['locs'])
    plots[x][y].set_title(PLOT_STYLES[p])


# Plot scores
f, plots = plt.subplots(3, 4)
for p in range(len(PLOT_STYLES)):
    y = p%4
    x = p/4
    print x, y
    plots[x][y].hist(results['100000']['scores'])
    plots[x][y].set_title(PLOT_STYLES[p])
