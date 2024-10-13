package hr.fer.oprpp1.custom.scripting.nodes;

import hr.fer.oprpp1.custom.collections.ArrayIndexedCollection;
import hr.fer.oprpp1.custom.scripting.elems.*;

/**
 * The class represents a tag starting with the key word "FOR".
 * 
 * @author Ivan Rep
 */
public class ForLoopNode extends Node {

	private ElementVariable variable;
	private Element startExpression;
	private Element endExpression;
	private Element stepExpression;

	/**
	 * The constructor method assigns an ArrayIndexedCollection of children, one element of type ElementVariable and three 
	 * elements of type Element.
	 * 
	 * @param arr the ArrayIndexedCollection object that contains the children of this node
	 * @param variable the element that represents a variable name
	 * @param startExpression the second element, it represents the starting expression
	 * @param endExpression the third element, it represents the end expression
	 * @param stepExpression the fourth element,  it represents the step expression
	 */
	public ForLoopNode(ArrayIndexedCollection arr, ElementVariable variable, Element startExpression, Element endExpression, Element stepExpression) {
		super(arr);
		this.variable = variable;
		this.startExpression = startExpression;
		this.endExpression = endExpression;
		this.stepExpression = stepExpression;
	}

	/**
	 * The constructor method assigns an ArrayIndexedCollection of children, one element of type ElementVariable and two
	 * elements of type Element.
	 * 
	 * @param arr the ArrayIndexedCollection object that contains the children of this node
	 * @param variable the element that represents a variable name
	 * @param startExpression the second element, it represents the starting expression
	 * @param endExpression the third element, it represents the end expression
	 */
	public ForLoopNode(ArrayIndexedCollection arr, ElementVariable variable, Element startExpression, Element endExpression) {
		super(arr);
		this.variable = variable;
		this.startExpression = startExpression;
		this.endExpression = endExpression;
	}

	/**
	 * The getter method for the element that represents the variable.
	 * 
	 * @return the ElementVariable object that represents the variable
	 */
	public ElementVariable getVariable() {
		return variable;
	}

	/**
	 * The getter method for the element that represents the starting expression.
	 * 
	 * @return the Element object that represents the starting expression
	 */
	public Element getStartExpression() {
		return startExpression;
	}

	/**
	 * The getter method for the element that represents the end expression.
	 * 
	 * @return the Element object that represents the end expression
	 */
	public Element getEndExpression() {
		return endExpression;
	}

	/**
	 * The getter method for the element that represents the step expression.
	 * 
	 * @return the Element object that represents the step expression
	 */
	public Element getStepExpression() {
		return stepExpression;
	}

	/**
	 * The method returns a String representation of the elements inside of a for loop tag.
	 * 
	 * @return the String that contains the variable name, the start expression, the end expression, and the step expression
	 * if provided
	 */
	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(" {$ FOR ").append(variable.asText()+" ").append(startExpression.asText()+" ").append(endExpression.asText()+" ");
		if( stepExpression!=null ) sb.append(stepExpression.asText()+" ");
		sb.append(" $} ");

		for(int i=0; i<this.numberOfChildren(); i++) {
			sb.append( ((Node) arr.get(i)).toString() );
		}

		sb.append("{$END$}");

		return sb.toString();
	}


	/**
	 * The method checks whether the current ForLoopNode object's and another node's 
	 * children are equal.
	 * 
	 * @return true if equal, otherwise false
	 */
	@Override
	public boolean equals(Object obj) {

		if (obj == this) return true; 

		if (!(obj instanceof ForLoopNode)) return false; 

		ForLoopNode node = (ForLoopNode) obj; 

		if(this.numberOfChildren()!=node.numberOfChildren()) return false;

		for(int i=0; i<this.numberOfChildren(); i++) {
			if( !this.getChild(i).equals( node.getChild(i) ) )
				return false;
		}

		boolean retValue = this.getReturnValue(node);

		return retValue;
	}

	/**
	 * The method compares the equality of the current ForLoopNode object's
	 * variables and the given ForLoopNode object's variables.
	 * 
	 * @param node the given node
	 * @return true if each of the string representations of the elements without any
	 * blank characters are equal, otherwise false
	 */
	public boolean getReturnValue(ForLoopNode node) {
		
		return removeBlanks( this.getVariable().asText() ).
				equals( removeBlanks( node.getVariable().asText() ) ) &&
				
				removeBlanks( this.getStartExpression().asText() ).
				equals( removeBlanks( node.getStartExpression().asText() ) ) && 
				
				removeBlanks( this.getEndExpression().asText() ).
				equals( removeBlanks( node.getEndExpression().asText() ) ) &&
				
				stepExpression==null ? true : 
					removeBlanks( this.getStepExpression().asText() ).
					equals( removeBlanks( node.getStepExpression().asText() ) ) ;
	}
	


}
