#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

typedef double (*PtrfunDouble)();

typedef struct
{
    void *vTable;
    int lowerBound;
    int upperBound;
} UnaryFunction;

typedef struct
{
    void *vTable;
    int lowerBound;
    int upperBound;
} Square;

typedef struct
{
    void *vTable;
    int lowerBound;
    int upperBound;
    double a;
    double b;
} Linear;

double valueAtSquare(Square *s, double x)
{
    return x * x;
}

double valueAtLinear(Linear *l, double x)
{
    return l->a * x + l->b;
}

double negativeValueAt(UnaryFunction *f, double x)
{
    return -(*(PtrfunDouble *)f->vTable)(x);
}

PtrfunDouble squareVTable[2] = {&valueAtSquare, &negativeValueAt};
PtrfunDouble linearVTable[2] = {&valueAtLinear, &negativeValueAt};

bool sameFunctionsForInts(UnaryFunction *f1, UnaryFunction *f2, double tolerance)
{
    if (f1->lowerBound != f2->lowerBound)
        return 0;
    if (f1->upperBound != f2->upperBound)
        return 0;

    for (int x = f1->lowerBound; x <= f1->upperBound; x++)
    {
        double delta = (*(PtrfunDouble *)f1->vTable)(f1, (double)x) -
                       (*(PtrfunDouble *)f2->vTable)(f2, (double)x);
        if (delta < 0)
            delta = -delta;
        if (delta > tolerance)
            return 0;
    }

    return 1;
}

void tabulate(UnaryFunction *f)
{
    for (int x = f->lowerBound; x <= f->upperBound; x++)
    {
        printf("f(%d)=%lf\n", x, (*(PtrfunDouble *)f->vTable)(f, (double)x));
    }
}

void constructSquare(Square *s, int lb, int ub)
{
    s->lowerBound = lb;
    s->upperBound = ub;
    s->vTable = &squareVTable;
}

void constructLinear(Linear *l, int lb, int ub, double a, double b)
{
    l->lowerBound = lb;
    l->upperBound = ub;
    l->vTable = &linearVTable;
    l->a = a;
    l->b = b;
}

Square *createSquare(int lb, int ub)
{
    Square *s = malloc(sizeof(Square));
    constructSquare(s, lb, ub);
    return s;
}

Linear *createLinear(int lb, int ub, double a, double b)
{
    Linear *l = malloc(sizeof(Linear));
    constructLinear(l, lb, ub, a, b);
    return l;
}

int main()
{
    UnaryFunction *f1 = createSquare(-2, 2);
    tabulate(f1);

    UnaryFunction *f2 = createLinear(-2, 2, 5, -2);
    tabulate(f2);

    printf("f1==f2: %s\n", sameFunctionsForInts(f1, f2, 1E-6) ? "DA" : "NE");
    printf("neg_val f2(1) = %lf\n", (*((PtrfunDouble *)f2->vTable)[1])(f2, 1.0));

    free(f1);
    free(f2);
}