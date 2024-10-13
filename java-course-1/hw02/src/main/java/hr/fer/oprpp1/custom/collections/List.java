package hr.fer.oprpp1.custom.collections;

/**
 * The interface is used for implementing the ArrayIndexedCollection and LinkedListIndexedCollection.
 * 
 * @author Ivan Rep
 */
public interface List extends Collection {

	/**
	 * This abstract method should get the object at the given index.
	 * 
	 * @param index the index of the object
	 * @return the object at the given index
	 */
	Object get(int index);
	
	/**
	 * This abstract method should insert an object with the given value at the given position.
	 * 
	 * @param value the value of the object that is added at the given position
	 * @param position the position where the object is added
	 */
	void insert(Object value, int position);
	
	/**
	 * This abstract method should find the index of the object with the given value.
	 * 
	 * @param value the object's value
	 * @return the index of the object with the given value
	 */
	int indexOf(Object value);
	
	/**
	 * This abstract method should remove the object at the given index.
	 * 
	 * @param index the index of the obejct to be removed
	 */
	void remove(int index);

}
