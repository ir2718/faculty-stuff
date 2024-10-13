package hr.fer.oprpp1.custom.scripting.elems;

/**
 * The class represents a constant Integer number element.
 * 
 * @author Ivan Rep
 */
public class ElementConstantInteger extends Element {

	private int value;
	
	/**
	 * The constructor assigns an integer value to the element
	 * 
	 * @param value the int value assigned to the element
	 */
	public ElementConstantInteger(int value) {
		this.value = value;
	}
	
	/**
	 * The method returns a String representation of the element
	 *
	 * @return the assigned value in the form of a String
	 */
	@Override
	public String asText() {
		return String.valueOf(value);
	}
	
}
