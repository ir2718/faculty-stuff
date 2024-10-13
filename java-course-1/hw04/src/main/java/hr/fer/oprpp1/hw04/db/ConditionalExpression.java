package hr.fer.oprpp1.hw04.db;

/**
 * The class represents a conditional
 * expression that consists of an attribute,
 * an operator and a literal string.
 * 
 * @author Ivan Rep
 */
public class ConditionalExpression {

	private IFieldValueGetter fieldGetter;
	private String stringLiteral;
	private IComparisonOperator comparisonOperator;

	/**
	 * The constructor method that sets all the values of a conditional expression.
	 * 
	 * @param fieldGetter the getter for the needed attribute
	 * @param stringLiteral the string literal 
	 * @param comparisonOperator the operator that compares the fieldGetter 
	 * and stringLiteral
	 */
	public ConditionalExpression(IFieldValueGetter fieldGetter, String stringLiteral, IComparisonOperator comparisonOperator) {
		this.fieldGetter = fieldGetter;
		this.stringLiteral = stringLiteral;
		this.comparisonOperator = comparisonOperator;
	}


	/**
	 * The getter method for the field getter.
	 * 
	 * @return the field getter
	 */
	public IFieldValueGetter getFieldGetter() {
		return fieldGetter;
	}

	/**
	 * The getter method for the string literal.
	 * 
	 * @return the string literal
	 */
	public String getStringLiteral() {
		return stringLiteral;
	}

	/**
	 * The getter method for the comparison operator.
	 * 
	 * @return the comparison operator
	 */
	public IComparisonOperator getComparisonOperator() {
		return comparisonOperator;
	}

	
}
