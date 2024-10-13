import itertools
import sys
import numpy as np

def makegen(f):
  return ( np.array([float(c) 
    for c in line[1:-2].split(',')])
      for line in itertools.takewhile(lambda x: x != "\n", f))

def read_data(file_path, read_from_file=True):
    if read_from_file:
        f = open(file_path)
    else:
        f = sys.stdin

    line = f.readline().split('),(')
    line[0] = line[0][7:]
    line[2] = line[2][:-3]
    P = np.array([[float(x)  for x in row.split(',')] for row in line])
    f.readline()

    qas = np.array(list(makegen(f)))
    qbs = np.array(list(makegen(f)))

    return P, qas, qbs    