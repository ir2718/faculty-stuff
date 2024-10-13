from sys import argv
from math import ceil

def read_file(filename):
    f = open(filename, 'r')
    lines = []
    [lines.append([float(num) for num in l.strip().split(' ')]) for l in f]
    f.close()
    return lines

def main():
    if(len(argv) != 2):
        exit('The program takes in 1 argument: path to file')

    l = read_file(argv[1])
    
    s = 'Hyp#Q10#Q20#Q30#Q40#Q50#Q60#Q70#Q80#Q90\n'
    for k, nums in enumerate(l):
        nums.sort()
        indices = [ceil(i/10*len(nums))-1 for i in range(1, 10)]
        s += '{:03d}#'.format(k+1)
        s += '#'.join([str(nums[index]) for index in indices])
        s += '\n'
    print(s)

main()