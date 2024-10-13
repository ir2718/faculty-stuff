package hr.fer.zemris.java.gui.layouts;

import java.awt.Component;

/**
 * Class represents a position in the calculator layout.
 * 
 * @author Ivan Rep
 */
public class RCPosition {
	
	private int row;
	private int column;
	
	/**
	 * Constructor. Sets the row and column values.
	 * 
	 * @param row		the given row value
	 * @param column	the given column value
	 */
	public RCPosition(int row, int column){
		this.row = row;
		this.column = column;
	}

	/**
	 * Getter method for the row count.
	 * 
	 * @return row count for the current RCPosition
	 */
	public int getRow() {
		return row;
	}
	
	/**
	 * Getter method for the column count.
	 * 
	 * @return column count for the current RCPosition
	 */
	public int getColumn() {
		return column;
	}
	
	/**
	 * Parses the text into an RCPosition object.
	 * 
	 * @param text 	the given text that is parsed
	 * @return		a new RCPosition object
	 */
	public static RCPosition parse(String text) {
		text = text.trim();
		String[] s = text.split(",");

		for(int i=0; i<s.length; i++) 
			s[i] = s[i].trim();
		
		if(s.length!=2) throw new IllegalArgumentException("The method expects 2 numbers separated by a comma."); 
		
		int[] values = new int[s.length];
		
		for(int i = 0; i<values.length; i++)
			try { values[i] = Integer.parseInt(s[i]); } 
			catch( NumberFormatException exc ) {
				throw new IllegalArgumentException("The value "+s[i]+" is not a number but method takes in only numbers.");
			}
		
		return new RCPosition( values[0], values[1] );
	}

	/**
	 * Returns a string representation of the RCPosition.
	 * 
	 * @return string representation of the RCPosition
	 */
	@Override
	public String toString() {
		return this.getRow()+", "+this.getColumn();
	}
	
	/**
	 * Calculates and returns a hashcode for the current RCPosition object.
	 * 
	 * @return hashcode for the current RCPosition object
	 */
	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + column;
		result = prime * result + row;
		return result;
	}

	/**
	 * Checks whether the current RCPosition and the given object have the same row and column value.
	 * 
	 * @param obj 	the given object
	 * @return 		true if the given object and the current RCPosition have the same column and row values
	 * 				otherwise false
	 */
	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		RCPosition other = (RCPosition) obj;
		if (column != other.column)
			return false;
		if (row != other.row)
			return false;
		return true;
	}
	
	
}
