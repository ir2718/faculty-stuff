package zad1Bolji;

public class Rhomb implements Shape {

	private double side;
	private Point center;

	public Rhomb(double side, Point center) {
		this.side = side;
		this.center = center;
	}

	@Override
	public void drawShape() {
		System.out.println("in drawRhomb");
	}

	@Override
	public void moveShape(Point translation) {
		System.out.println("in moveRhomb");
		this.center.translate(translation);
		System.out.println(this.center);
	}

}
