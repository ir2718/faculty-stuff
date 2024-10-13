#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#include <winbase.h>
#include <windef.h>
#include "myfactory.h"

typedef struct Animal* (*PTRFUN)();

void* myfactory(char const* libname, char const* ctorarg) {

    char buffer[100];
    sprintf(buffer, "./%s.dll", libname);
    // printf("%s", buffer);
    
    HINSTANCE module = LoadLibrary(buffer);
    PTRFUN ptrfun = (PTRFUN)GetProcAddress(module, "create");

    return (*ptrfun)(ctorarg);
}