package hr.fer.oprpp1.custom.scripting.elems;

/**
 * The class represents a String element.
 * 
 * @author Ivan Rep
 */
public class ElementString extends Element {
			
	private String value;
	
	/**
	 * The constructor assigns a value for the String element.
	 * 
	 * @param value the String value that represents the element
	 */
	public ElementString(String value) {
		this.value = value;
	}
	
	/**
	 * The method returns a value assigned to the String element.
	 * 
	 * @return the symbol for the operator element
	 */
	@Override
	public String asText() {
		return value;
	}
	
}
