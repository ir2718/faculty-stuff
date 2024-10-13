import numpy as np

class ObjObject():

    def __init__(self):
        self.v = []
        self.f = []

    def parse(self, filename):

        for l in open(filename, 'r'):
            if (l.startswith('#')):
                continue

            elif (l.startswith('v')):
                values = [float(i) for i in l.split()[1:]]
                self.v.append(values)

            elif (l.startswith('f')):
                values = [int(i) for i in l.split()[1:]]
                self.f.append(values)
    
