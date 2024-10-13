from sys import argv

def read_all_matrices(filename):
    f = open(filename, 'r')
    lines = f.readlines()
    f.close()

    matrices = {}
    i, k = 0, 65
    for j, l in enumerate(lines):
        if(j == len(lines) - 1):
            matrices[chr(k)] = read_matrix(lines[i:j+1])
            k += 1
            break

        if (l.strip() == ''):
            matrices[chr(k)] = read_matrix(lines[i:j])
            k += 1 
            i = j + 1

    return matrices

def dot(matrix1, matrix2):
    first_v = list(matrix1.values())
    second_v = list(matrix2.values())

    if (len(first_v[0]) != len(second_v)):
        exit('Matrices can\'t be multiplied.')

    w1, w2, h2 = len(first_v), len(second_v), len(second_v[0])

    matrix_d = {}
    for i in range(w1):
        new_elem_list = []
        for j in range(h2):
            new_elem = 0
            for k in range(w2):
                new_elem += matrix1[i][k]*matrix2[k][j]
            new_elem_list.append(new_elem)
        matrix_d[i] = new_elem_list
    
    return matrix_d

def read_matrix(lines):
    matrix_d = {}
    it = 0

    for l in lines:
        if (it == 0):
            nums = l.strip().split(' ')
            row = int(nums[0])
            col = int(nums[1])
            for i in range(row):
                matrix_d[i] = [0]*col
            it += 1
            continue

        if (l == '\n'):
            break

        else:
            nums = l.strip().split(' ')
            if (len(nums) < 3): exit('Input file matrix dimensions don\'t match')
            i, j = int(nums[0]), int(nums[1])
            matrix_d[i][j] = int(nums[2])

    return matrix_d

def print_matrix(matrices):
    for name, matrix in matrices.items():
        print(name + ':')
        for k, v in matrix.items():
            s = '  '.join(['  {:5.2f}'.format(num) for num in v])
            print('  ' + s)

def save_matrix(filename, matrix_t):
    s = ''

    k, v = list(matrix_t[1].keys()), list(matrix_t[1].values())
    s += '{0} {1}\n'.format(len(k), len(v[0]))

    for k, v in matrix_t[1].items():
        for i, num in enumerate(v):
            if (num != 0):
                s += '{} {} {:.2f}\n'.format(k, i, num)

    f = open(filename, 'w')
    f.write(s.strip())
    print('Rezultat zapisan u datoteku '+filename)
    print(s)

def main():

    if(len(argv) != 3):
        exit('Wrong number of arguments. The script takes in two arguments.')

    filename_r = argv[1]
    filename_w = argv[2]

    matrices = {}
    matrices = read_all_matrices(filename_r)

    items = list(matrices.items())
    matrices[items[0][0] + '*' + items[1][0]] = dot(items[0][1], items[1][1])
    print_matrix(matrices)
    save_matrix(filename_w, list(matrices.items())[-1])

main()