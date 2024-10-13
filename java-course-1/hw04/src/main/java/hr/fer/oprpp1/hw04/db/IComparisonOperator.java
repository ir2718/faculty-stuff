package hr.fer.oprpp1.hw04.db;

/**
 * The interface represents a comparison operator 
 * needed for implementing a database query.
 * 
 * @author Ivan Rep
 */
public interface IComparisonOperator {

	/**
	 * The method must compare the two given values using a
	 * comparator.
	 * 
	 * @param value1 the first given string
	 * @param value2 the second given string
	 * @return value1 compared to value2 using a comparator
	 */
	public boolean satisfied(String value1, String value2);
	
}
