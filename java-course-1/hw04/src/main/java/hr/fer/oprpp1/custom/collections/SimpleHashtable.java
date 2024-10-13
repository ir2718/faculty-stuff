package hr.fer.oprpp1.custom.collections;

import static java.lang.Math.*;

import java.util.ConcurrentModificationException;
import java.util.Iterator;
import java.util.NoSuchElementException;

/**
 * The class is a simple representation of a hashing table.
 * It stores table entries with a key of type K and
 * value of type V
 * 
 * @author Ivan Rep
 *
 * @param <K> the type of the key parameter
 * @param <V> the type of the value parameter
 */
public class SimpleHashtable<K,V> implements Iterable<SimpleHashtable.TableEntry<K,V>> {

	private int size;
	private TableEntry<K, V>[] arr;
	private int modificationCount;

	/**
	 * The class is used as objects stored
	 * in the hash table.
	 * 
	 * @author Ivan Rep
	 *
	 * @param <K> the type of the key parameter
	 * @param <V> the type of the value parameter
	 */
	static class TableEntry<K, V> {
		private K key;
		private V value;
		private TableEntry<K, V> next;

		/**
		 * The constructor method that assigns the key,
		 * value and the reference to the next node.
		 * 
		 * @param key the key parameter
		 * @param value the value parameter
		 * @param next the next node
		 * @throws IllegalArgumentException if the key is null
		 */
		public TableEntry(K key, V value, TableEntry<K, V> next) {
			if(key == null) throw new IllegalArgumentException("The key can't be null.");
			this.key = key;
			this.value = value;
			this.next = next;
		}

		/**
		 * The getter method for the key parameter.
		 * 
		 * @return the value of the key parameter
		 */
		public K getKey() {
			return this.key;
		}

		/**
		 * The getter method for the value parameter.
		 * 
		 * @return the value of the value parameter
		 */
		public V getValue() {
			return this.value;
		}

		/**
		 * The setter method for the value parameter.
		 * 
		 * @param value the new value of the value parameter
		 */
		public void setValue(V value) {
			this.value = value;
		}

	}

	/**
	 * The constructor method for the hash table.
	 * Initializes 16 slots in the hash table.
	 */
	public SimpleHashtable() {
		this(16);
	}


	/**
	 * The constructor method for the hash table
	 * that sets the amount of slots to the first
	 * equal or higher power of two.
	 * 
	 * @param capacity the given capacity value
	 * @throws NullPointerException if the given capacity is less than 1
	 */
	@SuppressWarnings("unchecked")
	public SimpleHashtable(int capacity) {
		modificationCount = 0;

		if(capacity<1) throw new NullPointerException("The capacity can't be less than 1 but it is "+capacity+".");

		double initCapacity = log10(capacity)/log10(2); 

		if( initCapacity!=ceil(initCapacity) )
			capacity = (int) Math.pow(2, ceil(initCapacity));

		arr =  ( TableEntry<K, V>[]) new TableEntry[capacity];
		size = 0;

	}

	/**
	 * The method puts a new table entry to the hash table
	 * with the given key and value parameters. Returns
	 * null if there is no entry with the same key otherwise
	 * returns the old value of the entry.
	 * 
	 * @param key the key parameter of the new table entry
	 * @param value the value parameter of the new table entry
	 * @return <code>null</code> if there is no entry with the same key,
	 * otherwise returns the old value of the value parameter
	 */
	public V put(K key, V value) {
		if( key == null ) throw new NullPointerException("The table entry can't have a null key.");
		if( ( (double) size ) / ( (double) arr.length ) >= 0.75 )
			this.doubleCapacity();

		int slot = abs(key.hashCode()%arr.length);

		if(arr[slot]==null) {
			modificationCount++;
			size++;
			arr[slot] =  new TableEntry<>(key, value, null);
			return null;
		} 

		V retValue = null;
		TableEntry<K, V> node = arr[slot];

		// does the first node have the same key
		if(node.key.equals(key)) {		
			retValue = node.value;
			node.value = value;
			return retValue;
		}

		while( node.next!=null && !node.next.key.equals(key)  ) 
			node = node.next;

		// if there is no node with the same key
		if(node.next==null) {		
			modificationCount++;
			size++;
			TableEntry<K, V> entry = new TableEntry<>(key, value, null);
			node.next = entry;
			return null;
		}

		// if there is a node with the same key
		retValue = node.next.value;
		node.next.value = value;
		return retValue;
	}


	/**
	 * The method reallocates a new array with
	 * two times the amount of slots and 
	 * puts the old elements into it.
	 */
	@SuppressWarnings("unchecked")
	private void doubleCapacity() {
		TableEntry<K, V>[] current = this.toArray();
		int newCapacity = this.arr.length*2;
		this.arr = ( TableEntry<K, V>[]) new TableEntry[newCapacity];
		size = 0;
		for(TableEntry<K, V> element : current )
			this.put(element.key, element.value);
	}

	/**
	 * Returns the value of the entry with the given key there is such key,
	 * otherwise returns null. 
	 * 
	 * @param key the key of the entry searched in the hash table
	 * @return <code>null<\code> if there is no such key, otherwise the value parameter
	 * for the given key
	 */
	public V get(Object key) {
		TableEntry<K, V> node = arr[abs(key.hashCode()%arr.length)];

		if(node == null) return null;

		if(node.key.equals(key)) return node.value;
		while(node.next!=null) {
			if(node.next.key.equals(key)) return node.next.value;
			node = node.next;
		}


		return null;
	}


	/**
	 * The method returns the amount of table entries inside
	 * the current hash table.
	 * 
	 * @return the amount of table entries in the hash table
	 */
	public int size() {
		return this.size;
	}

	/**
	 * Checks whether the current hash table contains an entry with the
	 * given key.
	 * 
	 * @param key the value of the key parameter
	 * @return true if there is an entry with the same key, otherwise false
	 */
	public boolean containsKey(Object key) {
		return this.get(key) != null ? true : false;
	}

	/**
	 * Checks whether the current hash table contains an entry with the
	 * given value.
	 * 
	 * @param value the value of the value parameter
	 * @return true if there is an entry with the same value, otherwise false
	 */
	public boolean containsValue(Object value) {
		TableEntry<K, V> node = null;
		for(TableEntry<K, V> entry : arr) {
			if(entry != null) {
				node = entry;
				if( node.value.equals(value) ) return true;
				while(node!=null) {
					if(node.next.key.equals(value)) return true;
					node = node.next;
				}
			}
		}
		return false;
	}

	/**
	 * The method removes the entry with the given key if there is such
	 * and returns its value. If there is no entry with the given key returns null.
	 * 
	 * @param key the key parameter of the entry
	 * @return if there is an entry with the given key returns its value,
	 * otherwise returns <code>null</code>
	 */
	public V remove(Object key) {
		int slot = abs(key.hashCode()%arr.length);
		TableEntry<K, V> node = arr[slot];

		// slot is empty
		if(node==null) return null;

		TableEntry<K, V> prev = null;
		while( !node.key.equals(key) ) 
			if( node.next!=null ) {
				prev = node;
				node = node.next;
			}
			else break;

		// no given key
		if( node == null ) return null;

		// given key found
		V retValue = null;
		modificationCount++;
		size--;
		node.key = null;
		retValue = node.value;
		if ( arr[slot] == node ) {
			if(node.next!=null) arr[slot] = node.next;
			else arr[slot] = null;
		} else {
			prev.next = node.next;
		}
		node = null;

		return retValue;
	}

	/**
	 * Checks whether the current hash table contains no elements
	 * 
	 * @return true if the hash table has no elements, otherwise false
	 */
	public boolean isEmpty() {
		return this.size == 0;
	}


	/**
	 * The method returns the string representation of the hash table in the
	 * format: [ key1=value1, key2=value2, ..., keyX=valueX ]
	 * 
	 * @return a String object of entry key and values surrounded 
	 * by square brackets
	 */
	@Override
	public String toString() {
		if(size==0) return "[]";
		
		StringBuilder sb = new StringBuilder("[");
		TableEntry<K, V> node = null; 
		
		for(TableEntry<K, V> entry : arr) {
			if(entry != null) {
				node = entry;
				while(node!=null) {
					sb.append(node.key+"="+node.value+", ");
					node = node.next;
				}
			}
		}
		
		sb.setLength(sb.length()-2);
		return sb.append("]").toString();
	}

	/**
	 * The method returns an array of references to the table entries.
	 * 
	 * @return an array of references to the table entries
	 */
	@SuppressWarnings("unchecked")
	public TableEntry<K,V>[] toArray() {
		TableEntry<K, V>[] array = ( TableEntry<K, V> [] ) new TableEntry[size];
		TableEntry<K, V> node = null; 
		int i=0;
		for(TableEntry<K, V> entry : arr) {
			if(entry != null) {
				node = entry;
				while(node!=null) {
					array[i++] = node;
					node = node.next;
				}
			}
		}
		return array;
	}

	/**
	 * The method removes all elements from the hash table.
	 */
	public void clear() {
		for(int i=0, m=arr.length; i<m; i++ )
			arr[i] = null;
		modificationCount++;
		size = 0;
	}

	
	/**
	 * The method creates and returns a new iterator for the hash table.
	 * 
	 * @return an iterator for the hash table
	 */
	@Override
	public Iterator<TableEntry<K, V>> iterator() {
		return new IteratorImpl();
	}


	/**
	 * The class represents an iterator for a SimpleHashtable object.
	 * 
	 * @author Ivan Rep
	 */
	private class IteratorImpl implements Iterator<SimpleHashtable.TableEntry<K,V>> {

		private int elemsHandled;
		private int slot;
		private int indexInSlot;
		private TableEntry<K,V> last;
		private int savedModificationCount;

		/**
		 * The constructor method assigns the values of the current slot, the current index in the slot,
		 * the reference to the last used table entry and the amount of modifications.
		 */
		public IteratorImpl() {
			elemsHandled = 0;
			slot = 0;
			indexInSlot = 0;
			last = null;
			savedModificationCount = SimpleHashtable.this.modificationCount;
		}

		/**
		 * The method checks if there's a next element in the hash table and returns true if there are any,
		 * otherwise false.
		 * 
		 * @return true if there is a next element, false otherwise
		 * @throws ConcurrentModificationException if the iterator isn't exactly the same as the hash table
		 */
		public boolean hasNext() {
			if( savedModificationCount != modificationCount ) throw new ConcurrentModificationException("The iterator differs from the hash table.");
			return elemsHandled<size;
		}

		/**
		 * The method returns the next table entry in the form of an 
		 * object of type TableEntry from the current hash table.
		 * 
		 * @return a new object of type TableEntry that represents the next table entry
		 * @throws ConcurrentModificationException  if the iterator isn't exactly the same as the hash table
		 * @throws NoSuchElementException if there is no next element
		 */
		public TableEntry<K, V> next() {
			if( savedModificationCount != modificationCount ) throw new ConcurrentModificationException("The iterator differs from the hash table.");
			if( elemsHandled>=size ) throw new NoSuchElementException("There is no next element.");

			while( arr[slot] == null )
				slot++;

			if( indexInSlot == 0 ) last = arr[slot];
			else if ( last.next!=null ) last = last.next;

			if( last.next == null ) {
				slot++;
				indexInSlot=0;
			} else {
				indexInSlot++;
			}

			elemsHandled++;

			return last;
		}

		/**
		 * The method removes the current element from the iterator.
		 * 
		 * @throws IllegalStateException if the method is invoked on the same table entry twice or 
		 * before getting the first element in the hash table.
		 * @throws ConcurrentModificationException  if the iterator isn't exactly the same as the hash table
		 */
		public void remove() {
			if( last == null || last.key == null ) throw new IllegalStateException("The method remove can't be invoked on the same element twice or before getting the first element.");
			if( savedModificationCount != modificationCount ) throw new ConcurrentModificationException("The iterator differs from the hash table.");
			SimpleHashtable.this.remove(last.key);
			savedModificationCount++;
			elemsHandled--;
		}

	}

}
