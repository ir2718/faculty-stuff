package zad1Bolji;

public class Point {
	
	private int x;
	private int y;

	public Point(int x, int y) {
		this.x = x;
		this.y = y;
	}

	public void translate(Point other) {
		this.x += other.x;
		this.y += other.y;
	}
	
	@Override
	public String toString() {
		return "["+this.x+", "+this.y+"]";
	}
}
