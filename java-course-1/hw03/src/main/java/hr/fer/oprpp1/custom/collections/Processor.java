package hr.fer.oprpp1.custom.collections;

/**
 * The class models a processor.
 * 
 * @author Ivan Rep
 * 
 */
public interface Processor<T> {

	/**
	 * This abstract method should process the given object.
	 * 
	 * @param value an object that needs to be processed
	 */
	void process(T value);

}