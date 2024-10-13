def mymax(iterable, value=lambda i: i):
    max_key = next(iter(iterable))
    max_val = value(max_key)

    for some_key in iterable:
        if(value(some_key) > max_val):
            max_val = value(some_key)
            max_key = some_key
    
    return max_key

def main():
    str_list = ['a', 'abc', 'abcde', 'abc', 'ab']
    int_list = [1, 3, 5, 7, 4, 6, 9, 2, 0]
    char_list = ['a', 'k', 'z', '2', 'f', 'd', 'c', 'b']

    D={'burek':8, 'buhtla':5, 'slanac':3, 'krafna':4}

    print(mymax(str_list))
    print(mymax(int_list))
    print(mymax(char_list))

    print(mymax(D, lambda i: D.get(i)))

    people = [('Siniša','Šegvić'), ('Marko','Čupić'), ('Zoran','Kalafatić'), ('Jan', 'Šnajder')]
    print(mymax(people, lambda x: (x[1], x[0])))

main()
