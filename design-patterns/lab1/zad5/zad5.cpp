#include <stdio.h>
#include <iostream>

using namespace std;

class B
{
public:
  virtual int __cdecl prva() = 0;
  virtual int __cdecl druga(int) = 0;
};

class D : public B
{
public:
  virtual int __cdecl prva() { return 42; }
  virtual int __cdecl druga(int x) { return prva() + x; }
};

typedef int (*PTRFUN)();
typedef int (*PTRFUN2)(B *, int);

void function(B *pb)
{
  unsigned long long **l = (unsigned long long **)pb;
  cout << (*(PTRFUN)(**l))() << endl;
  cout << (*(PTRFUN2)(*(*l + 1)))(pb, 1) << endl;
}

int main(void)
{
  D d;
  function(&d);
}