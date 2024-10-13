package hr.fer.oprpp1.custom.collections;

/**
 * This interface is used for getting elements from a collection.
 * 
 * @author Ivan Rep
 */
public interface ElementsGetter<T> {
	
	/**
	 * This abstract method should check whether there are any more elements in the collection
	 * 
	 * @return true if there are more elements, otherwise false
	 */
	boolean hasNextElement();
	
	/**
	 * This abstract method should get the next element in the collection.
	 * 
	 * @return the next element in the collection
	 */
	T getNextElement();
	
	/**
	 * This abstract method should call the method process for the given processor
	 * 
	 * @param p this processors' method process is called
	 */
	default void processRemaining(Processor<? super T> p) {
		while( this.hasNextElement() )
			p.process( this.getNextElement() );
	}
	
}
