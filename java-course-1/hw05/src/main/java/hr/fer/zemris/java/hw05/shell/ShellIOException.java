package hr.fer.zemris.java.hw05.shell;

/**
 * Class represents a runtime exception for the shell.
 * 
 * @author Ivan Rep
 */
public class ShellIOException extends RuntimeException {

	private static final long serialVersionUID = 1L;

	/**
	 * Constructor method.
	 * Sets the message of the exception to the given message.
	 * 
	 * @param message the given message
	 */
	public ShellIOException( String message ) {
		super( message );
	}
	
}
