package hr.fer.oprpp1.hw02.prob1;

/**
 * The class that represents a token used in lexical analysis.
 * 
 * @author Ivan Rep
 */
public class Token {

	TokenType type;
	Object value;
	
	/**
	 * The constructor method sets the type of the token and its value.
	 * 
	 * @param type a value from the TokenType enum
	 * @param value the value of the token
	 */
	public Token(TokenType type, Object value) {
		this.type = type;
		this.value = value;
	}
	
	/**
	 * The getter method for the value of the token.
	 * 
	 * @return the value of the Token object
	 */
	public Object getValue() {
		return this.value;
	}
	
	/**
	 * The getter method for the type of the token.
	 * 
	 * @return the type of the Token object
	 */
	public TokenType getType() {
		return this.type;
	}

	/**
	 * The method checks whether the current object's and the given object's token types are equal.
	 * 
	 * @param o the given object
	 * @return true if the objects are equal, otherwise false
	 */
	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		Token other = (Token) obj;
		if (type != other.type)
			return false;
		return true;
	}
	
}
