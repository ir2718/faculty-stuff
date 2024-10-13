from os import listdir
from sys import argv
from re import match

def read_student_file(filename):
    f = open(filename, 'r', encoding='utf-8')
    d = {}
    for l in f:
        info = l.strip().split(' ')
        d[info[0]] = [info[1]+','+info[2], {}]
    return d

def read_lab_file_and_add_pts(filename, student_dict, lab_num_set):
    f = open(filename, 'r', encoding='utf-8')
    _, lab_num, _ = filename.split('.')[0].split('_')
    lab_num_set.add(int(lab_num))
    for l in f:
        info = l.strip().split(' ')
        pts = float(info[1])
        curr_dict = student_dict[info[0]][1]
        if (lab_num not in curr_dict):
            curr_dict[int(lab_num)] = pts
        else:
            print('Warning: lab'+lab_num+' points for student '+student_dict[info[0]][0]+' were already calculated.')

def main():

    if (len(argv) !=2 ):
        exit('The script takes in one argument: path to directory')

    student_dict = read_student_file(argv[1]+'/studenti.txt')

    labs_files = [f for f in listdir(argv[1]) if match(r'Lab_[0-9]{2}_[a-z]{1}[0-9]{2}', f)]

    lab_num_set = set()
    for i, f in enumerate(labs_files):
        read_lab_file_and_add_pts(argv[1] + '/' + f, student_dict, lab_num_set)

    s = '{:<15s}{:<30s}'.format('JMBAG', 'Prezime, Ime')
    for lab_num in lab_num_set:
        s += '{:<10s}'.format('L'+str(lab_num))
    s += '\n'
    
    for k, v in student_dict.items():
        s += '{:<15s}{:<30s}'.format(k, v[0])
        for lab_num in lab_num_set:
            if (lab_num in v[1]): 
                s += '{:<10s}'.format(str(v[1][lab_num]))
            else:
                s += '{:<10s}'.format('-')
        s += '\n'

    print(s)
main()