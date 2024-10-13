package hr.fer.oprpp1.custom.collections;

/**
 * A class that models a stack data structure.
 * 
 * @author Ivan Rep
 */
public class ObjectStack {

	private ArrayIndexedCollection col;

	/**
	 * The constructor creates an empty stack.
	 */
	public ObjectStack() {
		col = new ArrayIndexedCollection();
	}

	/**
	 * The method returns the amount of objects present on the stack.
	 * 
	 * @return the amount of objects on the stack
	 */
	public int size() {
		return col.size();
	}

	/**
	 * The method checks if the stack is empty.
	 * 
	 * @return true if it is empty, otherwise false
	 */
	public boolean isEmpty() {
		return col.isEmpty();
	}

	/**
	 * The method pushes an object on the top of the stack.
	 * 
	 * @param value an object with this value is added 
	 * @throws NullPointerException if the given value is null
	 */
	public void push(Object value) {
		if(value == null) throw new NullPointerException("The given value was null! You can only add non null values into the ObjectStack.");
		col.add(value);
	}

	/**
	 * The method removes an object from the top of the stack.
	 * 
	 * @return the object from the top of the stack
	 * @throws EmptyStackException if the stack is empty 
	 */
	public Object pop() {
		if(col.size() == 0) throw new EmptyStackException("You can't pop from the stack because it is empty!");
		Object obj = col.get(col.size()-1);
		col.remove(col.size()-1);
		return obj;
	}

	/**
	 * The method returns the object at the top of the stack without removing it.
	 * 
	 * @return the object at the top of the stack
	 */
	public Object peek() {
		if( col.size() == 0 ) throw new EmptyStackException("You can't peek because the stack is empty!");
		return col.get(col.size()-1);
	}

	/**
	 * The method removes all objects from the stack.
	 */
	public void clear() {
		col.clear();
	}

}
