package hr.fer.oprpp1.custom.collections;


/**
 * The class models a collection based on pairs of keys and values.
 * 
 * @author Ivan Rep
 *
 * @param <K> key parameter
 * @param <V> value parameter
 */
public class Dictionary<K, V> {

	private ArrayIndexedCollection<Pair<K, V>> arr;

	/**
	 * The class models a pair with a key and a value that
	 * is used in the Dictionary class.
	 * 
	 * @author Ivan Rep
	 *
	 * @param <K> the key parameter
	 * @param <V> the value parameter
	 */
	private static class Pair<K, V> {
		private K key;
		private V value;

		/**
		 * The constructor method sets the key and value parameters
		 * to the given values.
		 * 
		 * @param key the key parameter
		 * @param value the value parameter
		 */
		public Pair(K key, V value) {
			if(key==null) throw new IllegalArgumentException("The key can't be null");
			this.key = key;
			this.value = value;
		}

	}

	/**
	 * Empty constructor.
	 */
	public Dictionary() {
		arr = new ArrayIndexedCollection<>();
	}
	
	public int size() {
		return arr.size();
	}

	/**
	 * The method checks whether the dictionary
	 * is empty.
	 * 
	 * @return true if it's empty, otherwise false
	 */
	public boolean isEmpty() {
		return arr.size()==0;
	}

	/**
	 * Removes all the value from the current dictionary.
	 */
	public void clear() {
		arr.clear();
	}

	/**
	 * If there is already a pair with the same key, the method returns the current 
	 * value of the pair and replaces it with the new one. Otherwise
	 * adds a new pair to the end of the dictionary.
	 * 
	 * @param key the key of the pair
	 * @param value the value of the pair
	 * @return the value of the old pair, otherwise null
	 */
	public V put(K key, V value) {
		Pair<K, V> curr = getPair(key);
		V retVal = null;
		
		if(curr==null) {
			Pair<K, V> pair = new Pair<>(key, value);
			arr.add(pair);
			return retVal;
		}

		retVal = curr.value;
		curr.value = value;
		return retVal;
	}

	/**
	 * Returns the value with the given key from the dictionary.
	 * 
	 * @param key the key parameter
	 * @return if there is no such key returns null, otherwise returns the 
	 * value of the given key 
	 */
	public V get(Object key) {
		Pair<K, V> pair = getPair( key );
		return pair == null ? null : pair.value;
	}


	/**
	 * Returns the pair with the given key from the dictionary
	 * 
	 * @param key the key parameter
	 * @return if there is no such key returns null, otherwise returns the 
	 * pair with the given keys
	 */
	private Pair<K, V> getPair(Object key) {
		for(int i=0, m=arr.size(); i<m; i++)
			if(arr.get(i).key.equals(key))
				return arr.get(i);
		return null;
	}


	/**
	 * Removes the element of the dictionary with the given key.
	 * Returns the value that was in the pair before the replacement.
	 * 
	 * @param key the key parameter 
	 * @return if there is no pair with the same key parameter returns null, 
	 * otherwise returns the value of the pair with the given key
	 */
	public V remove(K key) {
		if(arr.size()==0) return null;

		for(int i=0, m=arr.size(); i<m; i++)
			if(arr.get(i).key.equals(key)) {
				V value = arr.get(i).value;
				arr.remove(i);
				return value;
			}

		return null;
	}


}
