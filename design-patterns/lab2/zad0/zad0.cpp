#include <iostream>
#include <assert.h>

  struct Point{
    int x; 
    int y;
  };

  void translatePoint(Point* first, Point other) {
      first->x += other.x;
      first->y += other.y;
  }

  void printPoint(Point p) {
      std::cout << "[" << p.x << ", " << p.y << "]" << std::endl;
  }

  struct Shape{
    enum EType {circle, square, rhomb};
    EType type_;
  };

  struct Circle{
     Shape::EType type_;
     double radius_;
     Point center_;
  };

  struct Square{
     Shape::EType type_;
     double side_;
     Point center_;
  };

  struct Rhomb{
     Shape::EType type_;
     double side_;
     Point center_;
  };

  void drawSquare(struct Square*){
    std::cerr <<"in drawSquare\n";
  }

  void drawCircle(struct Circle*){
    std::cerr <<"in drawCircle\n";
  }

  void drawRhomb(struct Rhomb*){
    std::cerr <<"in drawRhomb\n";
  }

  void drawShapes(Shape** shapes, int n){
    for (int i=0; i<n; ++i){
      struct Shape* s = shapes[i];

      switch (s->type_){

      case Shape::square:
        drawSquare((struct Square*)s);
        break;

      case Shape::circle:
        drawCircle((struct Circle*)s);
        break;

      case Shape::rhomb:
        drawRhomb((struct Rhomb*)s);
        break;

      default:
        assert(0); 
        exit(0);
      }

    }
  }

  void moveShapes(Shape** shapes, int n, Point translate[]){
      for (int i=0; i<n; ++i){
      struct Shape* s = shapes[i];

      switch (s->type_){

      case Shape::square:
        translatePoint(&(((struct Square*)s)->center_), translate[i]);
        printPoint(((struct Square*)s)->center_);
        break;

      case Shape::circle:
        translatePoint(&(((struct Square*)s)->center_), translate[i]);
        printPoint(((struct Square*)s)->center_);
        break;

      // case Shape::rhomb:
      //   translatePoint(&(((struct Rhomb*)s)->center_), translate[i]);
      //   printPoint(((struct Rhomb*)s)->center_);
      //   break;

      default:
        assert(0); 
        exit(0);
      }

    }
  }

  int main(){
    int n=5;
    Shape* shapes[n];
    shapes[0] = (Shape*)new Circle;
    ((Circle*)shapes[0])->center_ = {0, 0};
    shapes[0]->type_=Shape::circle;

    shapes[1]=(Shape*)new Square;
    ((Square*)shapes[1])->center_ = {1, 2};
    shapes[1]->type_=Shape::square;
    
    shapes[2]=(Shape*)new Square;
    ((Square*)shapes[2])->center_ = {1, -2};
    shapes[2]->type_=Shape::square;
    
    shapes[3]=(Shape*)new Circle;
    ((Circle*)shapes[3])->center_ =  {4, -4};
    shapes[3]->type_=Shape::circle;

    shapes[3]=(Shape*)new Circle;
    ((Circle*)shapes[3])->center_ =  {4, -4};
    shapes[3]->type_=Shape::circle;

    shapes[4]=(Shape*)new Rhomb;
    ((Rhomb*)shapes[4])->center_ =  {10, 2};
    shapes[4]->type_=Shape::rhomb;
    
    drawShapes(shapes, n);

    Point points[n];
    points[0] = {1, 1};
    points[1] = {0, 0};
    points[2] = {3, -4};
    points[3] = {-1, 9};
    points[4] = {-2, 3};
    moveShapes(shapes, n, points);

  }