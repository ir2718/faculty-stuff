from urllib.request import urlopen
from re import findall
from sys import argv

def main():

    if (len(argv) != 2):
        exit('The script takes in one argument: url of webpage')
    
    webpage = urlopen(argv[1])
    byte_page = webpage.read()
    str_page = byte_page.decode('utf-8')
    print('Source code of the page is :')
    print(str_page)
    print()

    print('The links on this page are:')
    [print(link.format(1)) for link in findall(r'href="([^"]+)', str_page)]
    print()

    print('List of all hosts and number of refs from which the site can be accessed:')
    d = {}
    for host in findall(r'https{0,1}://([^/"\']+)',str_page):
        d[host] = 1 if host not in d else d[host] + 1
    [print('{0} ---> {1}'.format(str(it[0]), str(it[1]))) for it in d.items()]
    print()


    regex = '^(\w|\.|\_|\-)+[@](\w|\_|\-|\.)+[.]\w{2,3}$'
    print('The emails on this page are:')
    [print(link) for link in findall(regex, str_page)]
    print()

    print('Amount of links to pictures:')
    print(len(findall(r'<img.*src="([^"]+)', str_page)))
    print()

main()