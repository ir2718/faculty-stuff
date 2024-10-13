package hr.fer.oprpp1.custom.scripting.elems;

/**
 * The class represents a function element.
 * 
 * @author Ivan Rep
 */
public class ElementFunction extends Element {
	
	private String name;
	
	/**
	 * The constructor assigns a name for the function element
	 * 
	 * @param name the String value that represents the name of the function
	 */
	public ElementFunction(String name) {
		this.name = name;
	}
	
	/**
	 * The method returns a String representation of the function element.
	 * 
	 * @return the name of the function
	 */
	@Override
	public String asText() {
		return name;
	}
	
}
