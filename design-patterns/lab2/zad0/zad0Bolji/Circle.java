package zad1Bolji;

public class Circle implements Shape {

	private double radius;
	private Point center;

	public Circle(double radius, Point center) {
		this.radius = radius;
		this.center = center;
	}

	@Override
	public void drawShape() {
		System.out.println("in drawCircle");
	}

	@Override
	public void moveShape(Point translation) {
		System.out.println("in moveCircle");
		this.center.translate(translation);
		System.out.println(this.center);
	}

}