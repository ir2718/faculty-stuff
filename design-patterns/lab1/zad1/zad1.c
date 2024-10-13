#include <stdio.h>
#include <stdlib.h>

typedef char const *(*PTRFUN)();

typedef struct
{
  void *table;
  char *name;
} Animal;

char const *dogGreet(void)
{
  return "vau!";
}

char const *dogMenu(void)
{
  return "kuhanu govedinu";
}

char const *catGreet(void)
{
  return "mijau!";
}

char const *catMenu(void)
{
  return "konzerviranu tunjevinu";
}

PTRFUN dogTable[2] = {&dogGreet, &dogMenu};
PTRFUN catTable[2] = {&catGreet, &catMenu};

void constructDog(Animal *dog, char *name)
{
  dog->name = name;
  dog->table = &dogTable;
}

void constructCat(Animal *cat, char *name)
{
  cat->name = name;
  cat->table = &catTable;
}

Animal *createDog(char *name)
{
  Animal *dog = malloc(sizeof(Animal));
  constructDog(dog, name);
  return dog;
}

Animal *createCat(char *name)
{
  Animal *cat = malloc(sizeof(Animal));
  constructCat(cat, name);
  return cat;
}

void animalPrintGreeting(Animal *animal)
{
  printf("%s pozdravlja: %s\n", animal->name,  (*((PTRFUN *)animal->table)[0])());
}

void animalPrintMenu(Animal *animal)
{
  printf("%s voli %s\n", animal->name, (*((PTRFUN *)animal->table)[1])());
}

void testAnimals(void)
{
  struct Animal *p1 = createDog("Hamlet");
  struct Animal *p2 = createCat("Ofelija");
  struct Animal *p3 = createDog("Polonije");

  animalPrintGreeting(p1);
  animalPrintGreeting(p2);
  animalPrintGreeting(p3);

  animalPrintMenu(p1);
  animalPrintMenu(p2);
  animalPrintMenu(p3);

  free(p1);
  free(p2);
  free(p3);
}

void createOnHeap(void)
{
  Animal *p1 = createDog("Hamlet");
  animalPrintGreeting(p1);
  animalPrintMenu(p1);
  free(p1);
}

void createOnStack(void)
{
  Animal p1;
  constructDog(&p1, "Hamlet");
  animalPrintGreeting(&p1);
  animalPrintMenu(&p1);
}

void createNDogsMalloc(int n)
{
  Animal *p1 = malloc(n * sizeof(Animal));

  for (int i = 0; i < n; i++)
  {
    char *buffer = malloc(sizeof(char) * 10);
    sprintf(buffer, "%d", i);
    constructDog(p1 + i, buffer);
  }

  for (int i = 0; i < n; i++)
  {
    animalPrintGreeting(p1 + i);
    animalPrintMenu(p1 + i);
  }
}

int main(void)
{
  // 1)
  testAnimals();

  // 2)
  // createOnStack();
  // createOnHeap();

  //3)
  // int n = 7;
  // createNDogsMalloc(n);
}
