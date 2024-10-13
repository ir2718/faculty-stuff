#include <stdio.h>
#include <stdlib.h>
typedef char const* (*PTRFUN)();

typedef struct Tiger{
  PTRFUN* vtable;
  char* name;
} Tiger;

char const* name(void* this) {
    return ((Tiger*)this)->name;
}

char const* greet() {
    return "roar";
}

char const* menu() {
    return "meso";
}

PTRFUN* vtableTiger[3] = {name, greet, menu};

Tiger* create(char* name) {
    Tiger* p = malloc(sizeof(Tiger));
    construct(p, name);
    return p;
}

void construct(Tiger* p, char* name) {
    p->vtable = vtableTiger;
    p->name = name;
}