#include <stdio.h>
#include <string.h>

const void *mymax(const void *base, size_t nmemb, size_t size,
                  int (*compar)(const void *, const void *))
{
    char *base_char = (char*)base;
    void *max = base;
    for (int i = 1; i < nmemb; i++){
        void* other = base_char + size * i;
        int status = (*(compar))(other, max);
        if (status)
            max = other;
    }

    return max;
}

int gt_int(const int *a, const int *b)
{
    return *a > *b;
}

int gt_char(const char *a, const char *b)
{
    return *a > *b;
}

int gt_str(const char **a, const char **b)
{
    return strcmp(*a, *b) > 0;
}

int main()
{
    int arr_int[] = {1, 3, 5, 7, 4, 6, 9, 2, 0};
    char arr_char[] = "Suncana strana ulice";
    const char *arr_str[] = {"Gle", "malu", "vocku", "poslije", "kise",
                             "Puna", "je", "kapi", "pa", "ih", "njise"};

    int *max_int;
    max_int = (int *)mymax((void *)arr_int, 9, sizeof(int), gt_int);
    printf("%d\n", *max_int);

    char *max_char;
    max_char = (char *)mymax((void *)arr_char, 21, sizeof(char), gt_char);
    printf("%c\n", *max_char);

    char **max_str;
    max_str = (char **)mymax((void*)arr_str, sizeof(arr_str)/sizeof(char*), sizeof(char *), gt_str);
    printf("%s\n", *max_str);
}