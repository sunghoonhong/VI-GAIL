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
  dir ='%s/' % name
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
  max_mean = max(metrics[1])
  mean_mean = np.mean(metrics[1])
  mean_std = np.std(metrics[1])
  
  threshold = -10
  start_ep = np.argwhere(metrics[1] > threshold)
  start_ep = start_ep[0][0] if len(start_ep) > 0 else -1
  if start_ep > -1:
    new_metric = metrics[1][start_ep:] 
    new_mm = np.mean(new_metric)
    new_std = np.std(new_metric)
  else:
    new_mm = new_std = -1
  return max_mean, mean_mean, mean_std, start_ep * n, new_mm, new_std 

if __name__ == '__main__':
    root_dir = ''
    csv_path = '../total_stat.csv'
    parser = ArgumentParser()
    parser.add_argument('--n',      type=int, default=50)
    parser.add_argument('--recent', type=int, default=0)
    args = parser.parse_args()
    files= []
    stats= []
    for f in os.listdir():
      if f[-4:] == '.csv':
        files.append(f[:-4])
    files = sorted(files)
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
                
        maxm, mm, ms, start_ep, new_mm, new_std= drawall(name, episodes, metrics, labels, n=args.n, recent=args.recent)
        if start_ep == -1000:
            start_ep = '-'
        stat = [name, maxm, '%.2f#%.2f' % (mm, ms), start_ep, '%.2f#%.2f' %(new_mm, new_std) if new_std != -1 else '-']
        stats.append(stat)
            
        focus = '###' if maxm>=0.9999 else ''
        print(name, sum(timesteps), '%.2f'%maxm, '%.2f'%mm, '%.2f'%ms, start_ep, '%.2f' % new_mm, '%.2f' % new_std, focus)
      except Exception as e:
        print(str(e))
    with open(csv_path, 'a', newline='') as f:
        wrt = csv.writer(f)
        for row in stats:
            wrt.writerow(row)
            