package hr.fer.oprpp1.custom.collections;

import java.util.Arrays;

/**
 * The class models a double linked list with an object value inside each node.
 * 
 * @author Ivan Rep
 *
 */
public class LinkedListIndexedCollection extends Collection {

//	public static void main(String[] args)  {
//		ArrayIndexedCollection col = new ArrayIndexedCollection(2);
//		col.add(Integer.valueOf(20));
//		col.add("New York");
//		col.add("San Francisco"); // here the internal array is reallocated to 4
//		System.out.println(col.contains("New York")); // writes: true
//		col.remove(1); // removes "New York"; shifts "San Francisco" to position 1
//		System.out.println(col.get(1)); // writes: "San Francisco"
//		System.out.println(col.size()); // writes: 2
//		col.add("Los Angeles");
//		LinkedListIndexedCollection col2 = new LinkedListIndexedCollection(col);
//		// This is local class representing a Processor which writes objects to System.out
//		class P extends Processor {
//			public void process(Object o) {
//				System.out.println(o);
//			}
//		};
//		System.out.println("col elements:");
//		col.forEach(new P());
//		System.out.println("col elements again:");
//		System.out.println(Arrays.toString(col.toArray()));
//		System.out.println("col2 elements:");
//		col2.forEach(new P());
//		System.out.println("col2 elements again:");
//		System.out.println(Arrays.toString(col2.toArray()));
//		System.out.println(col.contains(col2.get(1))); // true
//		System.out.println(col2.contains(col.get(1))); // true
//		col.remove(Integer.valueOf(20)); // removes 20 from collection (at position 0).
//	}

	private int size;
	private ListNode first;
	private ListNode last;

	/**
	 * The getter method for the first node.
	 * 
	 * @return the first node of the list if it isn't empty, otherwise null
	 */
	public ListNode getFirst() {
		return this.first;
	}

	/**
	 * The getter method for the last node.
	 * 
	 * @return the last node of the list if it isn't empty, otherwise null
	 */
	public ListNode getLast() {
		return this.last;
	}

	/**
	 * This class models a node in a linked list.
	 * 
	 * @author Ivan Rep
	 *
	 */
	private static class ListNode {
		private Object value;
		private ListNode next;
		private ListNode previous;

		/**
		 * The constructor assigns a value to the ListNode object.
		 * 
		 * @param value the value assigned to the object
		 */
		ListNode(Object value){
			this.value = value;
		}
	}

	/**
	 * Empty constructor.
	 */
	public LinkedListIndexedCollection() {

	}

	/**
	 * The constructor copies the elements from the given collection into the list.
	 * 
	 * @param other the reference to the given collection
	 * @throws NullPointerException if the given collection is empty
	 */
	public LinkedListIndexedCollection(Collection other) {
		addAll(other);
	}

	/**
	 * The method returns the amount of ListNode objects in the list.
	 * 
	 * @return the amount of ListNode objects in the list
	 */
	@Override
	public int size() {
		return size;
	}

	/**
	 * The method returns an array with the same values as the nodes in the list.
	 * 
	 * @return an array of type Object with the same values as the nodes
	 */
	@Override
	public Object[] toArray() {
		Object[] obj = new Object[size];
		int i = 0;
		ListNode node = first;

		while(i<size) {
			obj[i++] = node.value;
			node = node.next;
		}

		return obj;
	}

	/**
	 * The method checks whether an object is in the list.
	 * 
	 * @return returns the index of the object if the list contains a node with the same value, otherwise -1
	 */
	@Override
	public boolean contains(Object value) {
		ListNode node = first;

		while(node!=null) {
			if(node.value.equals(value)) return true;
			node = node.next;
		}

		return false;
	}

	/**
	 * The method returns a String representation of the list.
	 * 
	 * @return String representation of the list
	 */
	@Override
	public String toString() {
		if(first==null) return "";

		StringBuilder sb = new StringBuilder();

		ListNode node = first;
		sb.append(first.value).append(", ");

		while(node.next!=null) {
			sb.append(node.next.value).append(", ");
			node = node.next;
		}

		return sb.substring(0, sb.length()-2).toString();
	}

	/**
	 * The method processes each element of the collection.
	 * 
	 * @param processor the processor whose methods are used for processing the elements
	 */
	public void forEach(Processor processor) {
		ListNode node = first;
		processor.process(node.value);
		while(node.next!=null) {  
			processor.process(node.next.value);
			node = node.next;
		}

	}


	/**
	 * The method adds the given value at the end of the list.
	 * 
	 *  @param value the value that is added to the end of the list
	 *  @throws NullPointerException if the given value is null
	 */
	public void add(Object value) {
		if(value == null) throw new NullPointerException("The given value is null! You can't add null values to the linked list.");

		ListNode node = new ListNode(value);
		if(first == null && last == null) {
			first = node;
			last = node;
		} else {
			last.next = node;
			node.previous = last;
			last = node;
		}
		size++;

	}

	/**
	 * The method deletes all the values from the list.
	 */
	public void clear() {
		first = null;
		last = null;
		size = 0;
	}


	/**
	 * The method returns the value of the node at the given index.
	 * 
	 * @param index the index of object
	 * @return the value of the object with the given value if it's in the list, otherwise -1
	 * @throws IndexOutOfBoundsException if the given index is less than 0 or bigger than size-1
	 */
	public Object get(int index) {
		if(index<0 || index>size-1) throw new IndexOutOfBoundsException("The given index should be between 0 and "+(size-1)+"! It was "+index+".");

		int i;
		ListNode node;

		// if the element is in the first half of the collection
		if(index<size/2) {
			node = first;
			i=0;
			while(i<index)  {
				node = node.next;
				i++;
			}
			return node.value;
		} 

		// if the element is in the second half of the collection
		node = last;
		i=size-1;
		while(i>index) {
			node = node.previous;
			i--;
		}

		return node.value;
	}


	/**
	 * The method insert a node with the given value at the given position. It moves all other nodes with a larger index one place towards 
	 * the end of the list.
	 * 
	 * @param value the value that is added to the list
	 * @param position the position at which the value is added
	 * @throws IndexOutOfBoundsException if the given position is less than 0 or larger than size 
	 * @throws NullPointerException if the given value is null
	 */
	public void insert(Object value, int position) {
		if(position<0 || position>size) throw new IndexOutOfBoundsException("The given position should be between 0 and "+size+"! It was "+position+".");

		this.add(value);
		if(position==size-1) return;

		int i = size-1;
		ListNode helper = last.previous;

		while(i>position) {
			helper.next.value = helper.value;
			if(i!=position+1) helper = helper.previous;
			i--;
		}

		helper.value = value;

	}

	/**
	 * The method finds and returns the index of the node with the given value.
	 * 
	 * @param value the value that is searched for
	 * @return the index of the node with the given value if it's in the list, otherwise -1
	 */
	public int indexOf(Object value) {
		if(size==0) return -1;

		ListNode node = first;
		int i=0;

		if(node.value.equals(value)) return i;
		i++;

		while(node.next!=null) {
			if(node.next.value.equals(value)) return i;
			node = node.next;
			i++;
		}

		return -1;
	}

	/**
	 * The method removes a node at the specified index.
	 * 
	 * @param index the index of the node which is to be removed
	 * @throws IndexOutOfBoundsException if the given index is less than 0 or larger than size-1
	 */
	public void remove(int index) {
		if(index<0 || index>size-1) throw new IndexOutOfBoundsException("The given index should be between 0 and "+(size-1)+"! It was "+index+".");

		if( size==1 && index==0 ) {
			clear();
			return;
		}

		ListNode node;
		size--;

		if(index==0) {
			node=first;
			first = first.next;
			first.previous = null;
			node = null;
			return;
		}

		if(index == size) { 
			node = last;
			last = last.previous;
			last.next = null;
			node = null;
			return;
		}

		// any other value between
		node = first;
		while(index>0) {
			node = node.next;
			index--;
		}

		ListNode node2 = node.previous;
		node2.next = node.next;
		node2 = node.next;
		node2.previous = node.previous;
		node = null;

	}

}
