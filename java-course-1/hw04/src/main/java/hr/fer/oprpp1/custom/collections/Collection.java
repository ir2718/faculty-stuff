package hr.fer.oprpp1.custom.collections;

/**
 * The interface represents a general collection of objects.
 * 
 * @author Ivan Rep
 *
 */
public interface Collection<T> {

	/**
	 * This default method determines whether the collection is empty.
	 * 
	 * @return true if empty, otherwise false
	 * @throws NullPointerException if the reference to the given collection is <code>null</code>
	 */
	default boolean isEmpty() {
		return this.size()==0;
	}

	/**
	 * This abstract method should calculate the size of the collection.
	 * 
	 * @return size of collection
	 */
	int size();

	/**
	 * This abstract method should add the given object into the collection.
	 * 
	 * @param value the value that is added to the collection
	 */
	void add(T value);

	/**
	 * This abstract method should determine whether the collection contains the given object.
	 * 
	 * @param value the method searches for the value in the collection
	 * @return true if there is an element with the given value, false otherwise
	 */
	boolean contains(Object value);

	/**
	 * This abstract method should remove the given object from the collection.
	 * 
	 * @param value the value of the object that needs to be removed
	 * @return true if the list contained the object, otherwise false
	 */
	boolean remove(Object value);

	/**
	 * This abstract method should allocate a new array with size equal to the size of this collection,
	 * fill it with the elements of the collection and return the array.
	 * 
	 * @return an array of objects of type T
	 */
	Object[] toArray();

	/**
	 * This abstract method should process each element of the collection.
	 * 
	 * @param processor the processor whose methods are used for processing the elements
	 */
	default void forEach(Processor<? super T> processor) {
		ElementsGetter<T> getter = createElementsGetter();
		getter.processRemaining(processor);
	}

	/**
	 * This default method adds all elements from the given collection into the current collection.
	 * The other collection is unchanged.
	 * 
	 * @param other this collection is added to the original collection
	 */
	default void addAll(Collection<? extends T> other) {

		/**
		 * The class models a processor.
		 * 
		 * @author Ivan Rep
		 * 
		 */
		class LocalProcessor implements Processor<T>{

			/**
			 * The method processes the given object.
			 * 
			 * @param value an object that needs to be processed
			 */
			@Override
			public void process(T value) {
				Collection.this.add(value);
			}

		}

		other.forEach(new LocalProcessor());

	}


	/**
	 * This abstract method should remove all elements from the collection.
	 */
	void clear();

	/**
	 * This abstract method should create and return an element getter for this type of collection.
	 * 
	 * @return the getter for this type of element
	 */
	ElementsGetter<T> createElementsGetter();

	/**
	 * This abstract method should add all elements from the given collection that are validated
	 * using a tester object to the current collection.
	 * 
	 * @param col the collection that is added to the current collection
	 * @param tester the tester used for validating the objects of the collection
	 */
	default void addAllSatisfying(Collection<T> col, Tester<? super T> tester) {
		ElementsGetter<T> getter = col.createElementsGetter();
		while(getter.hasNextElement()) {
			T obj = getter.getNextElement();
			if(tester.test(obj)) this.add(obj);
		}

	}


}
