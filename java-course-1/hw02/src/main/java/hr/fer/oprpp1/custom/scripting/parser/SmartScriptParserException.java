package hr.fer.oprpp1.custom.scripting.parser;

import hr.fer.oprpp1.custom.scripting.lexer.SmartScriptLexerException;

/**
 * The class represents an exception for whenever something is wrong while parsing 
 * the given input.
 * 
 * @author Ivan Rep
 */
public class SmartScriptParserException extends RuntimeException {

	private static final long serialVersionUID = 1L;

	/**
	 * The constructor method for the SmartScriptParserException.
	 * It assigns the message that is printed once thrown.
	 * 
	 * @param message the message that is printed once thrown
	 */
	public SmartScriptParserException(String message) {
		super(message);
	}
	
	/**
	 * The constructor method for the SmartScriptParserException.
	 * It assigns the given SmartScriptLexerException object's message
	 * to this object.
	 * 
	 * @param exc this object's message will be printed once thrown
	 */
	public SmartScriptParserException(SmartScriptLexerException exc) {
		super(exc.getMessage());
	}
	
}
