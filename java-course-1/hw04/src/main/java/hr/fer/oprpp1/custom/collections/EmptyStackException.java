package hr.fer.oprpp1.custom.collections;

/**
 * The class represents an exception for performing operations on an empty stack. 
 * It inherits RuntimeException.
 * 
 * @author Ivan Rep
 */
public class EmptyStackException extends RuntimeException {

	private static final long serialVersionUID = 1L;

	/**
	 * The method prints the given error on the console.
	 * 
	 * @param message the message that is printed on the console
	 */
	public EmptyStackException(String message) {
		super(message);
	}
}
