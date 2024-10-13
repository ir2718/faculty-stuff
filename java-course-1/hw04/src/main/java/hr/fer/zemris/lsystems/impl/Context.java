package hr.fer.zemris.lsystems.impl;

import hr.fer.oprpp1.custom.collections.ObjectStack;

/**
 * The class represents a context
 * of a stack used in creating and
 * drawing an L-system.
 * 
 * @author Ivan Rep
 */
public class Context {

	private ObjectStack<TurtleState> stack;
	
	/**
	 * The empty constructor creates a new stack.
	 */
	public Context() {
		stack = new ObjectStack<>();
	}

	/**
	 * The method gets the state from
	 * the top of the stack.
	 * 
	 * @return a TurtleState type object 
	 */
	public TurtleState getCurrentState() {
		return stack.peek();
	}

	/**
	 * The method pushes a TurtleState type object
	 * on the top of the stack.
	 * 
	 * @param state the state that is pushed
	 */
	public void pushState(TurtleState state) {
		stack.push(state);
	}

	/**
	 * The method pops the state from the top of the stack.
	 */
	public void popState() {
		stack.pop();
	}
}
