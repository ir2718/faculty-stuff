package hr.fer.zemris.java.gui.charts;

import java.util.ArrayList;
import java.util.List;

import static java.lang.Math.*;

/**
 * Class used for representing a bar chart.
 * 
 * @author Ivan Rep
 */
public class BarChart {
	
	private List<XYValue> list;
	private String descX; 
	private String descY; 
	private int yMin; 
	private int yMax; 
	private int diffY;
	
	/**
	 * Constructor. Sets the given values.
	 * 
	 * @param list 	given list of XYValues
	 * @param descX given description for x axis
	 * @param descY given description for y axis
	 * @param yMin	given minimal value of y
	 * @param yMax	given maximum value of y
	 * @param diffY	given difference on y axis
	 */
	public BarChart( List<XYValue> list, String descX, String descY, int yMin, int yMax, int diffY ) {
		if( list.stream().filter( p -> p.getY() > yMin ).count() == 0 )
			 throw new IllegalArgumentException("The list can't containt a point that has a smaller y value that yMin.");
		
		this.list = list;
		this.descX = descX;
		this.descY = descY;
		
		if (yMin<0 ) throw new IllegalArgumentException("MinY can't be negative. Yours was"+yMin+".");
		this.yMin = yMin;
		
		if( yMax<yMin ) throw new IllegalArgumentException("MaxY can't be less than minY.");
		this.yMax = yMax;
		
		if( diffY<0 ) throw new IllegalArgumentException("DiffY can't be less than zero.");
		this.diffY = (yMax-yMin) % diffY == 0 ? diffY : (int) (diffY + (yMax-yMin)%diffY);
	}

	/**
	 * Returns the list of XYValues for the bar chart.
	 * 
	 * @return list of XYValues
	 */
	public List<XYValue> getList() {
		return list;
	}

	/**
	 * Returns the description of x axis for the bar chart.
	 * 
	 * @return description of x axis 
	 */
	public String getDescX() {
		return descX;
	}

	/**
	 * Returns the description of y axis for the bar chart.
	 * 
	 * @return description of y axis 
	 */
	public String getDescY() {
		return descY;
	}

	/**
	 * Returns the minimum value for y for the bar chart.
	 * 
	 * @return minimum value for y
	 */
	public int getyMin() {
		return yMin;
	}

	/**
	 * Returns the maximum value for y for the bar chart.
	 * 
	 * @return maximum value for y
	 */
	public int getyMax() {
		return yMax;
	}

	/**
	 * Returns the difference for the y value for the bar chart.
	 * 
	 * @return difference for the y value
	 */
	public int getDiffY() {
		return diffY;
	}
	
	

} 
