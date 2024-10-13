package hr.fer.oprpp1.custom.collections;

/**
 * The class represents a general collection of objects.
 * 
 * @author Ivan Rep
 *
 */
public class Collection {
	
	/**
	 * Empty constructor for the collection.
	 */
	protected Collection() {

	}

	/**
	 * The method determines whether the collection is empty.
	 * 
	 * @return true if empty, otherwise false
	 * @throws NullPointerException if the reference to the given collection is <code>null</code>
	 */
	public boolean isEmpty() {
		return this.size() == 0;
	}

	/**
	 * The method calculates the size of the collection.
	 * 
	 * @return size of collection
	 */
	public int size() {
		return 0;
	}

	/**
	 * The method adds the given object into the collection.
	 * 
	 * @param value the value that is added to the collection
	 */
	public void add(Object value) {

	}

	/**
	 * The method determines whether the collection contains the given object.
	 * 
	 * @param value the method searches for the value in the collection
	 */
	public boolean contains(Object value) {
		return false;
	}

	/**
	 * The method removes the given object from the collection.
	 * 
	 * @param value the value of the object that needs to be removed
	 * @return true if the list contained the object, otherwise false
	 */
	public boolean remove(Object value) {
		return false;
	}

	/**
	 * The method allocates a new array with size equal to the size of this collection,
	 * fills it with the elements of the collection and returns the array.
	 * 
	 * @throws UnsupportedOperationException whenever the method is called
	 */
	public Object[] toArray() {
		throw new UnsupportedOperationException();
	}

	/**
	 * The method processes each element of the collection.
	 * 
	 * @param processor the processor whose methods are used for processing the elements
	 */
	public void forEach(Processor processor) {
		
	}

	/**
	 * The method adds all elements from the given collection into the current collection.
	 * The other collection is unchanged.
	 * 
	 * @param other
	 */
	public void addAll(Collection other) {
		
		/**
		 * The class models a processor.
		 * 
		 * @author Ivan Rep
		 * 
		 */
		class LocalProcessor extends Processor{

			/**
			 * The method processes the given object.
			 * 
			 * @param value an object that needs to be processed
			 */
			@Override
			public void process(Object value) {
				Collection.this.add(value);
			}

		}
		
		other.forEach(new LocalProcessor());

	}


	/**
	 * The method removes all elements from the collection.
	 */
	public void clear() {

	}

}
