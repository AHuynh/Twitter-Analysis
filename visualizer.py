import datetime as dt
import matplotlib.pyplot as plt
import math
from matplotlib.ticker import StrMethodFormatter, NullFormatter
import numpy as np
import pandas as pd

########################################################################################
## Dataset visualization
dep_var = 'Favorites'

dataset = pd.read_csv('dataset.csv')
X = dataset.iloc[:, 1:-2]
y = dataset.iloc[:, -2 if dep_var == 'Favorites' else -1].values

def simple_x_vs_avg_faves(feature, dataset, size=100):
  buckets = dataset.groupby(dataset[feature])
  means = buckets[dep_var].mean()
  for key, item in buckets:
    print(key, len(buckets.get_group(key)))
  plt.title('Avg Faves VS ' + feature)
  plt.scatter(means.index, means, s=size)
  plt.show()
  plt.close()

########################################################################################
# Favorites VS date
year_buckets = dataset.groupby(dataset['Year'])
year_means = year_buckets[dep_var].mean()

times = []
for i in range(0, len(dataset)):
  tweet = dataset.iloc[i]
  times.append(dt.datetime(int(tweet['Year']), int(tweet['Month']), int(tweet['Day'])))
plt.title(dep_var + ' over years')
plt.yscale('log')
plt.scatter(times, y, s=0.5)
plt.scatter(pd.to_datetime(year_means.index, format='%Y'), year_buckets[dep_var].mean(), color='red')
ax = plt.gca()
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
ax.yaxis.set_minor_formatter(NullFormatter())
#plt.show()
plt.close()

########################################################################################
# Favorites VS sentiment (raw)
# -0.9|-0.7|-0.5|-0.3|-0.1|+0.1|+0.3|+0.5|+0.7|+0.9
sentiment_buckets = [[] for _ in range(11)]
# Also create some buckets and take the average of each bucket.
for s in range(0, len(dataset['Sentiment'].values)):
  sentiment = dataset['Sentiment'][s]
  sentiment_buckets[math.floor((sentiment+1)*5)].append(y[s])
sentiment_means = [np.mean(i) for i in sentiment_buckets]
plt.title(dep_var + ' VS Sentiment')
plt.yscale('log')
plt.scatter(dataset['Sentiment'].values, y, s=0.5)
plt.scatter([-1 + 0.2 * i for i in range(0, len(sentiment_means))], sentiment_means, color='red')
ax = plt.gca()
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
ax.yaxis.set_minor_formatter(NullFormatter())
plt.show()
plt.close()

########################################################################################
# Avg Faves VS day
dow_buckets = dataset.groupby(dataset['Day of week'])
dow_means = dow_buckets[dep_var].mean()
plt.title(dep_var + ' VS Day of Week')
plt.scatter(dow_means.index, dow_means)
ticks = list(range(0, 7))
labels = "Mon Tue Wed Thu Fri Sat Sun".split()
plt.xticks(ticks, labels)
#plt.show()
plt.close()

########################################################################################
# Avg Faves VS other columns
#simple_x_vs_avg_faves('Tweet Length', dataset, size=5)
#simple_x_vs_avg_faves('Is reply', dataset)
#simple_x_vs_avg_faves('User mentions', dataset)
#simple_x_vs_avg_faves('Hashtags', dataset)
#simple_x_vs_avg_faves('Pictures', dataset)
#simple_x_vs_avg_faves('Videos', dataset)