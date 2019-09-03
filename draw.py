import os
import csv
from argparse import ArgumentParser
import numpy as np
from matplotlib import pyplot as plt

def smooth(arr, n):
    end = -(len(arr)%n)
    if end == 0:
      end = None
    arr = np.reshape(arr[:end], (-1, n))
    arr = np.mean(arr, axis=1)
    return arr
  
def drawall(name, x, metrics, labels, n=100, recent=0):
  dir ='save_graph/%s/' % name
  if not os.path.exists(dir):
    os.makedirs(dir)
  
  x = smooth(x[-recent:], n)
  for i, metric in enumerate(metrics):
    metrics[i] = smooth(metric[-recent:], n)

  def draw(x, y, ylabel):
    plt.figure(figsize=(15,5))
    plt.plot(x, y)
    plt.xlabel('episode')
    plt.ylabel(ylabel)
    plt.savefig(dir+'/'+ylabel)
    plt.clf()
    

  for i, metric in enumerate(metrics):
    draw(x, metric, labels[i])

if __name__ == '__main__':
    root_dir = 'save_graph/'
    parser = ArgumentParser()
    parser.add_argument('--n',      type=int, default=50)
    parser.add_argument('--recent', type=int, default=0)
    args = parser.parse_args()
    files= []
    for f in os.listdir():
      if f[-4:] == '.csv':
        files.append(f[:-4])

    for name in files:
      filename = name + '.csv'
      if not os.path.exists(filename):
        continue
      
      episodes = []
      rewards = []
      scores = []
      timesteps = [] 
      pmaxs = []
  
      metrics = [
          rewards,
          scores,
          timesteps,
          pmaxs,
      ]
      labels = [
          'reward_sum',
          'score',
          'timestep',
          'pmax'
      ]
      try:
        with open(filename, 'r') as f:
            read = csv.reader(f)
            for i, row in enumerate(read):
                episodes.append(i)
                rewards.append(float(row[0]))
                scores.append(int(float(row[1])))
                timesteps.append(int(float(row[2])))
                pmaxs.append(float(row[3]))
                
        drawall(name, episodes, metrics, labels, n=args.n, recent=args.recent)
        print(name, sum(timesteps), max(scores))
      except Exception as e:
        print(str(e))