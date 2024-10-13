package hr.fer.oprpp1.custom.scripting.elems;

/**
 * The element represents a constant double number element.
 * 
 * @author Ivan Rep
 */
public class ElementConstantDouble extends Element {

	private double value;
	
	/**
	 * The constructor that assigns a double value to the element
	 * 
	 * @param value the double value assigned to the element
	 */
	public ElementConstantDouble(double value) {
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
