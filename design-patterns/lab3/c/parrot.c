#include <stdio.h>
#include <stdlib.h>
typedef char const* (*PTRFUN)();

typedef struct Parrot{
  PTRFUN* vtable;
  char* name;
} Parrot;

char const* name(void* this) {
    return ((Parrot*)this)->name;
}

char const* greet() {
    return "squawk";
}

char const* menu() {
    return "sjemenke";
}

PTRFUN* vtableParrot[3] = {name, greet, menu};

Parrot* create(char* name) {
    Parrot* p = malloc(sizeof(Parrot));
    construct(p, name);
    return p;
}

void construct(Parrot* p, char* name) {
    p->vtable = vtableParrot;
    p->name = name;
}