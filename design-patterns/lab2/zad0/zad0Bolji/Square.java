package zad1Bolji;

public class Square implements Shape {

	private double side;
	private Point center;

	public Square(double side, Point center) {
		this.side = side;
		this.center = center;
	}

	@Override
	public void drawShape() {
		System.out.println("in drawSquare");
	}

	@Override
	public void moveShape(Point translation) {
		System.out.println("in moveSquare");
		this.center.translate(translation);
		System.out.println(this.center);
	}

}