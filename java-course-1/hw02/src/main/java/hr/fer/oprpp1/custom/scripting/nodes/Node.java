package hr.fer.oprpp1.custom.scripting.nodes;

import hr.fer.oprpp1.custom.collections.ArrayIndexedCollection;

/**
 * The abstract class provides a general node element in a generative tree.
 * 
 * @author Ivan Rep
 */
public abstract class Node {
	
	ArrayIndexedCollection arr;
	
	/**
	 * The constructor method assigns an ArrayIndexedCollection object that contains 
	 * the children of the node.
	 * 
	 * @param arr the ArrayIndexedCollection object that contains the children of the node
	 */
	public Node(ArrayIndexedCollection arr) {
		this.arr = arr;
	}
	
	/**
	 * The method adds the child to the ArrayIndexedCollection object of the node.
	 * 
	 * @param child the child node that is added
	 */
	public void addChild(Node child) {
		arr.add(child);
	}
	
	/**
	 * The method returns the number of direct children of the current node.
	 * 
	 * @return the number of direct children of the current node
	 */
	public int numberOfChildren() {
		return arr.size();
	}

	/**
	 * The getter method for the child of a node.
	 * 
	 * @param index the index of the child in the ArrayIndexedCollection object
	 * @return the child with the given index
	 * @throws IndexOutOfBoundsException if the index is larger than size-1 or less than 0
	 */
	public Node getChild(int index) {
		if( index>arr.size()-1 || index<0 ) throw new IndexOutOfBoundsException("The index is larger than size-1 or less than 0.");
		return (Node) arr.get(index);
	}
	
	/**
	 * The method returns the given string without any blank characters.
	 * It is used while checking the equality of two nodes.
	 * 
	 * @param str the string whose blank characters are removed
	 * @return the string wihout any blank characters
	 */
	public static String removeBlanks(String str) {
		return str.replace("\\n", "\n").replace(" ", "").replace("\n", "").replace("\r", "").replace("\t", "");
	}
	
}
