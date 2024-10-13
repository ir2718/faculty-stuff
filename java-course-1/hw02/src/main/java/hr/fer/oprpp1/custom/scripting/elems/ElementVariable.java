package hr.fer.oprpp1.custom.scripting.elems;

/**
 * The class that represents a variable element.
 * 
 * @author Ivan Rep
 */
public class ElementVariable extends Element {

	private String name;
	
	/**
	 * The constructor that assigns a name to the variable element.
	 * 
	 * @param name the name assigned to the element
	 */
	public ElementVariable(String name) {
		this.name = name;
	}
	
	/**
	 * The method returns a name assigned to the variable element.
	 * 
	 * @return the name of the variable element
	 */
	@Override
	public String asText() {
		return name;
	}
	
	
}
