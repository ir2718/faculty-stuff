package hr.fer.oprpp1.hw04.db;

/**
 * The class is a strategy design pattern
 * for the next comparison operators:
 * <, <=, >, <=, =, != and like.
 * 
 * @author Ivan Rep
 */
public class ComparisonOperators {
	
	/**
	 * Compares the first value to the second value using a <code><<code> comparator.
	 * 
	 * @param value1 the first string
	 * @param value2 the second string
	 * @return true if value1 is less than value2, otherwise false
	 */
	public static final IComparisonOperator LESS = ( value1, value2  ) -> value1.compareTo(value2)<0;
	
	/**
	 * Compares the first value to the second value using a <code><=<code> comparator.
	 * 
	 * @param value1 the first string
	 * @param value2 the second string
	 * @return true if value1 is less or equal to value2, otherwise false
	 */
	public static final IComparisonOperator LESS_OR_EQUALS = ( value1, value2 ) -> value1.compareTo(value2)<=0;
	
	/**
	 * Compares the first value to the second value using a <code>><code> comparator.
	 * 
	 * @param value1 the first string
	 * @param value2 the second string
	 * @return true if value1 is greater than value2, otherwise false
	 */
	public static final IComparisonOperator GREATER = ( value1, value2 ) -> value1.compareTo(value2)>0;
	
	/**
	 * Compares the first value to the second value using a <code>>=<code> comparator.
	 * 
	 * @param value1 the first string
	 * @param value2 the second string
	 * @return true if value1 is greater or equal to value2, otherwise false
	 */
	public static final IComparisonOperator GREATER_OR_EQUALS = ( value1, value2 ) -> value1.compareTo(value2)>=0;
	
	/**
	 * Compares the first value to the second value using a <code>==<code> comparator.
	 * 
	 * @param value1 the first string
	 * @param value2 the second string
	 * @return true if value1 is equal to value2, otherwise false
	 */
	public static final IComparisonOperator EQUALS = ( value1, value2 ) -> value1.compareTo(value2)==0;
	
	/**
	 * Compares the first value to the second value using a <code>!=<code> comparator.
	 * 
	 * @param value1 the first string
	 * @param value2 the second string
	 * @return true if value1 is not equal to value2, otherwise false
	 */
	public static final IComparisonOperator NOT_EQUALS = ( value1, value2 ) -> value1.compareTo(value2)!=0;
	
	/**
	 * Compares the first value to the second value using a like comparator.
	 * The asterisk sign is replaced with a arbitrary string.
	 * 
	 * @param value1 the first string
	 * @param value2 the second string
	 * @return true if value1 and value2's non asterisk part match
	 */
	public static final IComparisonOperator LIKE = ( value1, value2 ) -> {
		if ( value2.chars().filter( ch -> ch=='*' ).count() > 1 )  throw new IllegalArgumentException("A row contains more than one asterisk.");
		if ( value2.equals("*") ) return true;
		
		String[] arrAsterisk = value2.split("\\*");
		String first = arrAsterisk[0];
		
		if(arrAsterisk.length==1) return value1.startsWith(first);
		
		String second = arrAsterisk[1];
				
		return value1.substring(0, first.length()).startsWith(first) &&
				value1.substring(first.length()).endsWith(second);
	};
	
	
}
