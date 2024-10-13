package hr.fer.oprpp1.hw02.prob1;

/**
 * The class represents an exception that is used in the Lexer class.
 * 
 * @author Ivan Rep
 */
public class LexerException extends RuntimeException {

	private static final long serialVersionUID = 1L;

	/**
	 * The constructor that sets the value that's printed to the user.
	 * 
	 * @param message the message that is printed 
	 */
	public LexerException(String message) {
		super(message);
	}
	
}
