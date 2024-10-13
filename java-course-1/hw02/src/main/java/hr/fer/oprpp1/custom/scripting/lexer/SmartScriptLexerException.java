package hr.fer.oprpp1.custom.scripting.lexer;

/**
 * The class represents an exception that is used in the SmartScriptLexer class.
 * 
 * @author Ivan Rep
 */
public class SmartScriptLexerException extends RuntimeException {

	private static final long serialVersionUID = 1L;

	/**
	 * The constructor that sets the message that's printed to the user.
	 * 
	 * @param message the message that is printed 
	 */
	public SmartScriptLexerException(String message) {
		super(message);
	}
	
}