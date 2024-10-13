#include <iostream>
#include <cstring>
#include <set>
#include <vector>
#include <bits/stdc++.h>

using namespace std;

template <typename Iterator, typename Predicate>
Iterator mymax(Iterator first, Iterator last, Predicate pred)
{
  Iterator max = first;
  while (first != last)
  {
    if (pred(*first, *max))
      max = first;

    first++;
  }
  return max;
}

bool gt_int(const int a, const int b)
{
  return a > b;
}

bool gt_char(const char a, const char b)
{
  return a > b;
}

bool gt_str(string a, string b)
{
  return a > b;
}

int main()
{
  int arr_int[] = {1, 3, 5, 7, 4, 6, 9, 2, 0};
  char arr_char[] = "Suncana strana ulice";
  string arr_str[] = {"Gle", "malu", "vocku", "poslije", "kise",
                      "Puna", "je", "kapi", "pa", "ih", "njise"};
  vector<string> vec_str(arr_str, arr_str+sizeof(arr_str)/sizeof(arr_str[0]));
  set<string> set_str(arr_str, arr_str+sizeof(arr_str)/sizeof(arr_str[0]));

  const int *maxint = mymax(&arr_int[0], &arr_int[sizeof(arr_int) / sizeof(*arr_int)], gt_int);
  std::cout << *maxint << "\n";

  const char *maxchar = mymax(&arr_char[0], &arr_char[sizeof(arr_char) / sizeof(*arr_char)], gt_char);
  std::cout << *maxchar << "\n";

  const string *maxstr = mymax(&arr_str[0], &arr_str[sizeof(arr_str) / sizeof(*arr_str)], gt_str);
  std::cout << *maxstr << "\n";

  auto maxstrvec = mymax(vec_str.begin(), vec_str.end(), gt_str);
  std::cout << *maxstrvec << "\n";

  auto maxstrset = mymax(set_str.begin(), set_str.end(), gt_str);
  std::cout << *maxstrset << "\n";
}