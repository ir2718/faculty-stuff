package hr.fer.zemris.java.gui.charts;

/**
 * Class represents a point used for defining a bar chart.
 * 
 * @author Ivan Rep
 */
public class XYValue {
	
	private int x;
	private int y;
	
	/**
	 * Constructor. Sets the x and y values. 
	 * 
	 * @param x
	 * @param y
	 */
	public XYValue( int x, int y ) {
		this.x = x;
		this.y = y;
	}

	/**
	 * Getter method for the x value
	 * 
	 * @return x value of the current XYValue
	 */
	public int getX() {
		return x;
	}
	
	/**
	 * Getter method for the y value
	 * 
	 * @return y value of the current XYValue
	 */
	public int getY() {
		return y;
	}
	
}
