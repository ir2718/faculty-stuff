package zad1Bolji;

public class Main {

	public static void main(String[] args) {
		int n = 5;
		Shape[] shapes = new Shape[n];
		shapes[0] = new Circle(1.0, new Point(0, 0));
		shapes[1] = new Square(2.0, new Point(1, 2));
		shapes[2] = new Square(5.0, new Point(1, -2));
		shapes[3] = new Circle(7.0, new Point(4, -4));
		shapes[4] = new Rhomb(1.5, new Point(10, 2));
		drawShapes(shapes);

		Point[] points = new Point[n];
		points[0] = new Point(0, 0);
		points[1] = new Point(1, 1);
		points[2] = new Point(3, -4);
		points[3] = new Point(-1, 9);
		points[4] = new Point(-2, 3);
		System.out.println("------------------------");
		moveShapes(shapes, points);

	}

	private static void drawShapes(Shape[] shapes) {
		for(Shape shape : shapes) 
			shape.drawShape();
	}

	private static void moveShapes(Shape[] shapes, Point[] translations) {
		for(int i=0; i<shapes.length; i++) 
			shapes[i].moveShape(translations[i]);
	}

}

