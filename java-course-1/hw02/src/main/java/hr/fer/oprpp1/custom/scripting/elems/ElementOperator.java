package hr.fer.oprpp1.custom.scripting.elems;

/**
 * The class represents an operator element.
 * 
 * @author Ivan Rep
 */
public class ElementOperator extends Element {

	private String symbol;
	
	/**
	 * The constructor assigns the symbol for the operator element
	 * 
	 * @param symbol the String value that represents the symbol for the function
	 */
	public ElementOperator(String symbol) {
		this.symbol = symbol;
	}
	
	/**
	 * The method returns a String representation of the operator element.
	 * 
	 * @return the symbol for the operator element
	 */
	@Override
	public String asText() {
		return symbol;
	}
	
}
