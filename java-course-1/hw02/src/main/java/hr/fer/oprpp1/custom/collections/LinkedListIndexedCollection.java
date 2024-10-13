package hr.fer.oprpp1.custom.collections;

import java.util.ConcurrentModificationException;
import java.util.NoSuchElementException;

/**
 * The class models a double linked list with an object value inside each node.
 * 
 * @author Ivan Rep
 *
 */
public class LinkedListIndexedCollection implements List, Collection {

	public static void main(String[] args) {
		LinkedListIndexedCollection c = new LinkedListIndexedCollection();
		c.add(0);
		c.remove(0);
		System.out.println(c);
		c.add(1);c.add(2);c.add(3);
		System.out.println(c);
		ElementsGetter get = new LinkedListElementsGetter(c);
		get.getNextElement();
		c.remove(1);
		System.out.println(c);
		System.out.println(get.getNextElement());
		
	}

	private int size;
	private ListNode first;
	private ListNode last;
	private long modificationCount;

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
	public int size() {
		return size;
	}

	/**
	 * The method returns an array with the same values as the nodes in the list.
	 * 
	 * @return an array of type Object with the same values as the nodes
	 */
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
		modificationCount++;

	}

	/**
	 * The method deletes all the values from the list.
	 */
	public void clear() {
		first = null;
		last = null;
		size = 0;
		modificationCount++;
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
		
		this.add(value);	// the method add is called so there is no need for incrementing modificationCount
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
		modificationCount++;
		
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

	/**
	 * The method removes the object with the given value if present in the ArrayIndexedCollection.
	 * 
	 * @param index the index of the object that will be removed 
	 * @return true if successfully removed, otherwise false
	 */
	public boolean remove(Object value) {
		if(size == 0) return false;
		
		int index = indexOf(value);
		
		if(index==-1) return false;

		remove(index);	// the method remove is called so there is no need for incrementing modificationCount
		return true;
	}

	/**
	 * The method creates and returns a new object of type ElementsGetter for the 
	 * LinkedListIndexedCollection.
	 * 
	 * @return returns a new object of type ElementsGetter
	 */
	public ElementsGetter createElementsGetter() {
		return new LinkedListElementsGetter(this);
	}
	
	/**
	 * This class gets the elements one by one from the current LinkedListIndexedCollection. 
	 * 
	 * @author Ivan Rep
	 */
	private static class LinkedListElementsGetter implements ElementsGetter {

		private int currentIndex;
		private long savedModificationCount;
		private LinkedListIndexedCollection listRef;
		private ListNode ref;
		
		/**
		 * The constructor sets the currentIndex to 0 and sets the savedModificationCount equal to 
		 * the ref's modificationCount.
		 * 
		 * @param ref the reference to an LinkedListIndexedCollection object
		 */
		public LinkedListElementsGetter(LinkedListIndexedCollection listRef) {
			currentIndex = 0;
			savedModificationCount = listRef.modificationCount;
			ref = listRef.first;
			this.listRef = listRef;
		}
		
		/**
		 * The method checks whether the ElementsGetter object can get the next element in the
		 * current LinkedListIndexedCollection.
		 * 
		 * @return true if there are more elements, otherwise false
		 * @throws ConcurrentModificationException if the savedModificationcount and the modificationCount
		 * of the referenced collection aren't equal
		 */
		@Override
		public boolean hasNextElement() {
			if(savedModificationCount!=listRef.modificationCount) 
				throw new ConcurrentModificationException("The LinkedListIndexedCollection was changed!");
			
			return currentIndex < listRef.size();
		}

		/**
		 * The method returns the next element in the current LinkedListIndexedCollection.
		 * 
		 * @return the next element in the collection
		 * @throws ConcurrentModificationException if the savedModificationcount and the modificationCount
		 * of the referenced collection aren't equal
		 * @throws NoSuchElementException if the reference to the collection is null or if the collections'
		 * size is equal to the currentIndex
		 */
		@Override
		public Object getNextElement() {
			if( listRef==null || savedModificationCount!=listRef.modificationCount) 
				throw new ConcurrentModificationException("The LinkedListIndexedCollection was changed!");
			
			if( listRef==null || listRef.size()==currentIndex ) 
				throw new NoSuchElementException("There are no other elements in this collection!");
			
			Object ret = ref.value;
			ref = ref.next;
			currentIndex++;
			return ret;
		}
		
	}


}
