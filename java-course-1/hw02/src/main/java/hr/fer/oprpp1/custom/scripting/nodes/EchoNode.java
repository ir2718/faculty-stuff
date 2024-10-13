package hr.fer.oprpp1.custom.scripting.nodes;

import hr.fer.oprpp1.custom.collections.ArrayIndexedCollection;
import hr.fer.oprpp1.custom.scripting.elems.Element;
import hr.fer.oprpp1.custom.scripting.elems.ElementString;

/**
 * The class represents a tag starting with an "=" sign.
 * 
 * @author Ivan Rep
 */
public class EchoNode extends Node {

	private Element[] elements;
	
	/**
	 * The constructor method for the EchoNode.
	 * 
	 * @param arr an ArrayIndexedCollection object that contains the children of the echo node
	 * @param elem an array of objects of type Element inside the tag
	 */
	public EchoNode(ArrayIndexedCollection arr, Element... elem) {
		super(arr);
		int elemLen = elem.length;
		elements = new Element[elemLen];
		for(int i=0; i<elemLen; i++)
			elements[i] = elem[i];
	
	}

	/**
	 * The getter method for the array of elements inside the tag.
	 * 
	 * @return an array of elements inside the tag.
	 */
	public Element[] getElements() {
		return elements;
	}
	
	/**
	 * The method returns a String representation of the echo node.
	 * 
	 * @return the string representation of this node
	 */
	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(" {$= ");
		for(Element element : elements) {
			if(element instanceof ElementString) sb.append("\"" +element.asText().replace("\\n", "\n")+ "\"");
			else sb.append(element.asText()+" ");
		}
		sb.append("$} ");

		return sb.toString();
	}
	
	
	/**
	 * The method checks if the current EchoNode object and another node are
	 * both of the EchoNode type.
	 * 
	 * @param obj the second node
	 */
	@Override
	public boolean equals(Object obj) {
		
        if ( obj == this ) return true; 
        
        if ( !(obj instanceof EchoNode) ) return false; 
		
        EchoNode node = (EchoNode) obj; 
        
		return this.getReturnValue( node );
	}
	
	
	/**
	 * The method compares the equality of the current EchoNode object's
	 * variables and the given EchoNode object's variables.
	 * 
	 * @param node the given node
	 * @return true if each of the string representations of the elements without any
	 * blank characters are equal, otherwise false
	 */
	public boolean getReturnValue(EchoNode node) {

		for( int i=0; i<elements.length; i++) 
			if( !( removeBlanks( elements[i].asText() ) ).equals( removeBlanks( node.getElements()[i].asText() ) ) ) {
				System.out.println( removeBlanks( elements[i].asText() )+"------"+ removeBlanks( node.getElements()[i].asText() ) );
				return false;
			}
		
		return true;
	}
	
	
}
