package hr.fer.oprpp1.custom.collections;

/**
 * The interface is used for validating objects.
 * 
 * @author Ivan Rep
 */
public interface Tester<T> {

	/**
	 * This abstract method should check whether the object is acceptable.
	 * 
	 * @param obj the object that is checked
	 * @return returns true if it's acceptable, otherwise false
	 */
	boolean test(T obj);
	
}
