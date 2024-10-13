package hr.fer.oprpp1.custom.scripting.nodes;

import hr.fer.oprpp1.custom.collections.ArrayIndexedCollection;

/**
 * The class represents a node for the document in constructing the generative tree.
 * 
 * @author Ivan Rep
 */
public class DocumentNode extends Node {

	/**
	 * The constructor assigns the given collection of children to the document node.
	 * 
	 * @param arr the ArrayIndexedCollection object of children that is assigned
	 */
	public DocumentNode(ArrayIndexedCollection arr) {
		super(arr);
	}
	
	/**
	 * The method returns a String representation of the document node
	 * and recursively traverses it's children.
	 * 
	 * @return a String representation of the document.
	 */
	@Override
	public String toString() {
		
		StringBuilder sb = new StringBuilder();
		
		for(int i=0; i<this.numberOfChildren(); i++)
			sb.append( this.getChild(i).toString() );
		
		return sb.toString();
	}
	

	/**
	 * The method checks whether the generative trees of two documents
	 * are equal.
	 * 
	 * @param obj the second document
	 */
	@Override
	public boolean equals(Object obj) {
		
        if (obj == this) return true; 
        
        if (!(obj instanceof DocumentNode)) return false; 

        DocumentNode node = (DocumentNode) obj; 
        
        if(this.numberOfChildren()!=node.numberOfChildren()) return false;
          		
		for(int i=0; i<this.numberOfChildren(); i++) {
			if( !this.getChild(i).equals( node.getChild(i) ) )
				return false;
		}
		
		return true;
	}
	
	
	
}
