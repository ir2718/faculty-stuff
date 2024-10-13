package hr.fer.oprpp1.custom.scripting.nodes;

import hr.fer.oprpp1.custom.collections.ArrayIndexedCollection;

/**
 * The class represents a node for text that isn't in the tags.
 * 
 * @author Ivan Rep
 */
public class TextNode extends Node{

	private String text;
	
	/**
	 * The constructor method that assigns the children of the node, and the text
	 * that represents the node.
	 * 
	 * @param arr the ArrayIndexedCollection object that contains the children of the node
	 * @param text the text that represents the node
	 */
	public TextNode(ArrayIndexedCollection arr, String text) {
		super(arr);
		this.text = text;
	}
	
	/**
	 * The getter method for the textual representation of the node.
	 * 
	 * @return the text representation of the node
	 */
	public String getText() {
		return text;
	}
	
	/**
	 * The method returns a textual representation of the node.
	 * It returns the same String as the getText() method.
	 * 
	 * @return the text representation of the node
	 */
	@Override
	public String toString() {
		return text;
	}
	
	/**
	 * The method checks whether the current TextNode object
	 * and the given object are both of the same type.
	 * 
	 * @return true if they're both instanceof TextNode, otherwise false
	 */
	@Override
	public boolean equals( Object obj ) {
		
		if( this == obj ) return true;
		
		if( !(obj instanceof TextNode) ) return false;
		
		TextNode node = (TextNode) obj;
		
		return removeBlanks( this.getText() ).
				equals( removeBlanks( node.getText() ) );
	}
	
}
