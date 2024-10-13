package hr.fer.zemris.java.gui.layouts;

/**
 * Class represents an exception used in GUI.
 * 
 * @author Ivan Rep
 */
public class CalcLayoutException extends RuntimeException {
	
	private static final long serialVersionUID = 1L;

	/**
	 * Constructor. Sets the given message.
	 * 
	 * @param message the given message
	 */
	public CalcLayoutException(String message) {
		super(message);
	}
	
}
