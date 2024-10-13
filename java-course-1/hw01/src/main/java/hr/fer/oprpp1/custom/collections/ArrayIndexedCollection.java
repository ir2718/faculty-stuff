package hr.fer.oprpp1.custom.collections;

import static java.lang.Math.max;

import java.util.Arrays;

/**
 * The class models a collection that allows duplicate elements but doesn't allow <code>null</code> values
 * 
 * @author Ivan Rep
 *
 */
public class ArrayIndexedCollection extends Collection{

	private static final int CAPACITY = 16;

	private Object[] elements; 
	private int size;

	/**
	 * This constructor allocates the memory needed for 16 elements of type Object.
	 */
	public ArrayIndexedCollection() {
		this(CAPACITY);
	}

	/**
	 * This constructor allocates memory for the wanted amount of objects.
	 * 
	 * @param initialCapacity the amount of wanted objects in the array
	 * @throws IllegalArgumentException if the given initialCapacity is less or equal than 1
	 */
	public ArrayIndexedCollection(int initialCapacity) {
		if(initialCapacity < 1) throw new IllegalArgumentException("The parameter InitialCapacity must be 1 or larger! It was "+initialCapacity+".");
		elements = new Object[initialCapacity];
	}

	/**
	 * This constructor allocates memory for 16 objects or the amount of objects in the given collection, whichever is larger
	 * and copies the elements from the given collection to the current one.
	 * 
	 * @param other the collection whose elements will be copied
	 * @throws NullPointerException if the given reference is <code>null</code>
	 */
	public ArrayIndexedCollection(Collection other) {
		this( other, CAPACITY );
	}

	/**
	 * This constructor allocates memory for initialCapacity amount objects or the amount of objects in the given collection, 
	 * whichever is larger. It also copies the elements from the given collection to the current one.
	 * 
	 * @param other the collection whose elements will be copied
	 * @param initialCapacity if bigger than the size of the given collection, memory for this amount of objects will be allocated
	 * @throws NullPointerException if the given reference <code>other</code> is <code>null</code>
	 */
	public ArrayIndexedCollection(Collection other, int initialCapacity) {
		this( max(other.size(), initialCapacity) );
		this.addAll(other);
		size = other.size();
	}

	/**
	 * The method processes each element of the collection.
	 * 
	 * @param processor the processor whose methods are used for processing the elements
	 */
	@Override
	public void forEach(Processor processor) {
		for(Object obj : elements)
			if(obj!=null) processor.process(obj);
			else break;
	}

	/**
	 * The getter method for the length of the element array.
	 * 
	 * @return length of element array
	 */
	public int getElementsLength() {
		return elements.length;
	}

	/**
	 * The getter method for the elements array.
	 * 
	 * @return elements array
	 */
	@Override
	public Object[] toArray() {
		return elements;
	}

	/**
	 * Returns the size of the ArrayIndexedCollection object 
	 */
	@Override
	public int size() {
		return size;
	}
	
	/**
	 * The method checks whether the ArrayIndexedCollection contains the object with the given value.
	 * 
	 * @param an object with this value is searched for
	 * @return returns true if it's in the collection, otherwise false
	 */
	@Override
	public boolean contains(Object value) {
		for(int i=0, m=size; i<m; i++) 
			if(elements[i].equals(value)) 
				return true;
		
		return false;
	}

	/**
	 * Adds the given object into the collection in the first empty place.
	 * 
	 * @param value the object that is going to be added to the collection
	 * @throws NullPointerException if the given value is null
	 */
	@Override
	public void add(Object value) {
		if(value == null) throw new NullPointerException("The given value was null! You can only add non null values into the ArrayIndexedCollection.");

		if (elements.length==size) 
			elements = Arrays.copyOf(elements, 2*size);

		elements[size++] = value;
	}

	/**
	 * Returns the object that is stored in the array at the position index. 
	 * 
	 * @param index of the object that the method will return
	 * @throws IndexOutOfBoundsException if the given index is greater than size-1 or less than 1
	 */
	public Object get(int index) {
		if(index>size-1 || index<0) throw new IndexOutOfBoundsException("The index must be between 0 and "+(size-1)+"! The given index was "+index+".");
		return elements[index];	
	}

	/**
	 * Removes all elements from the collection. The allocated array is left at current capacity.
	 */
	@Override
	public void clear() {
		for(int i=0; i<size; i++)
			elements[i]=null;
		size = 0;
	}

	/**
	 * Insert an object with the given value into the array moving all the values with a larger index one element further
	 * 
	 * @param value an object with this value is inserted in the ArrayIndexedCollection
	 * @param position the object is inserted at this position
	 * @throws NullPointerException if the given value is null
	 * @throws IndexOutOfBoundsException if the position is bigger than size or less than 0
	 */
	public void insert(Object value, int position) {
		if(value == null) throw new NullPointerException("The given value was null! You can only add non null values into the ArrayIndexedCollection.");
		if(position>size || position<0) throw new IndexOutOfBoundsException("The position must be between 0 and "+size+"! The given position was "+position+".");

		if(elements.length-size==0)
			elements = Arrays.copyOf(elements, 2*size);

		for(int i=size-1; i>=position; i--)
			elements[i+1]=elements[i];

		size++;
		elements[position]=value;

	}

	/**
	 * Returns the index of the first occurrence of the given number or -1 if the value is not found.
	 * 
	 * @param value the value that the method tries to find in the array
	 * @return if it finds the value, returns its index, otherwise -1
	 */
	public int indexOf(Object value) {

		for(int i=0; i<size; i++) 
			if(value.equals(elements[i]))
				return i;

		return -1;
	}

	/**
	 * The method removes the object at the given index and shifts every other element with a larger index one place closer to the beginning of the array.
	 * 
	 * @param index the index of the object that will be removed 
	 * @throws IndexOutOfBoundsException if the given index is larger than size-1 or less than 0
	 */
	public void remove(int index) {
		if(index<0 || index>size-1) throw new IndexOutOfBoundsException("The index is must be between 0 and "+(size-1)+"! The given index was "+index+".");

		for(int m=size-1; index<m; index++)
			elements[index] = elements[index+1];

		elements[--size] = null;

	}

}
