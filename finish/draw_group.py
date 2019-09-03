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
  

if __name__ == '__main__':
    root_dir = 'group/'
    if not os.path.exists(root_dir):
      os.makedirs(root_dir)
    #csv_path = '../total_stat.csv'
    parser = ArgumentParser()
    parser.add_argument('--n',      type=int, default=50)
    parser.add_argument('--recent', type=int, default=0)
    args = parser.parse_args()
    files= []
    stats= []
    for f in os.listdir():
      if f[-4:] == '.csv' and '_LS' in f:
        files.append(f[:-4])
    files = sorted(files)
    plt.figure(figsize=(18,7))
    
    for i in range(0, len(files), 3):
      group_name = files[i].split('_LS')[0]
      ls_name = files[i] + '.csv'
      lsv_name = files[i+1] + '.csv'
      lsvns_name = files[i+2] + '.csv'
      group_files = [ls_name, lsv_name, lsvns_name]
      episodes = [[], [], []]
      scores = [[], [], []]
      
      for index, filename in enumerate(group_files):
        with open(filename, 'r') as f:
          read = csv.reader(f)
          for idx, row in enumerate(read):
            episodes[index].append(idx)
            scores[index].append(int(float(row[1])))
      min_len = min([len(episodes[0]), len(episodes[1]), len(episodes[2])])
      episodes = np.arange(min_len)
      episodes = smooth(episodes[-args.recent:], args.n)
      colors = ['r','g','b']
      
      for idx in range(3):
        scores[idx] = scores[idx][:min_len]
        scores[idx] = smooth(scores[idx][-args.recent:], args.n)
        plt.plot(episodes, scores[idx], colors[idx])
      ax = plt.gca()
      ax.xaxis.set_tick_params(labelsize=25)
      ax.yaxis.set_tick_params(labelsize=30)
      # ax.annotate('Episode', xy=(1, -0.08), ha='left', va='top', xycoords='axes fraction', fontsize='30')
      # ax.annotate('Score', xy=(-0.08, 1), xytext=(-15,2), ha='left', va='top',
                  # xycoords='axes fraction', textcoords='offset points',fontsize='30')
      plt.title(group_name, position=(0.5, 1.05), fontsize=40)
      plt.savefig(root_dir + group_name)
      plt.clf()
      
                