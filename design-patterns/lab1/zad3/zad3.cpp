#include<iostream>

using namespace std;

class CoolClass{
    public:
        virtual void set(int x){x_=x;};
        virtual int get(){return x_;};
    private:
        int x_;
};

class PlainOldClass{
  public:
    void set(int x){x_=x;};
    int get(){return x_;};
  private:
    int x_;
};

int main(void) {

  CoolClass c;
  cout << sizeof(c) << endl;

  PlainOldClass p;
  cout << sizeof(p) << endl;

}