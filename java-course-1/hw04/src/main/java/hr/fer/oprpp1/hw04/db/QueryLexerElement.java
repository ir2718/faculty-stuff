package hr.fer.oprpp1.hw04.db;

/**
 * The class represents an element used in lexing the database query.
 * 
 * @author Ivan Rep
 */
public class QueryLexerElement {
	
	private QueryLexerConstant type;
	private String value;
	
	/**
	 * The getter for the type of the lexer element.
	 * 
	 * @return a enum of type QueryLexerConstant
	 * representing the type
	 */
	public QueryLexerConstant getType() {
		return type;
	}

	/**
	 * The getter for the string representing the element.
	 * 
	 * @return the string representing the element
	 */
	public String getValue() {
		return value;
	}

	/**
	 * The constructor method that sets the type and value parameters.
	 * 
	 * @param type the type of the element
	 * @param value the string representation of the element
	 */
	public QueryLexerElement(QueryLexerConstant type, String value) {
		this.type = type;
		this.value = value;
	}

}
