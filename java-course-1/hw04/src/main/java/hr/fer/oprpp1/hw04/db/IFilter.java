package hr.fer.oprpp1.hw04.db;

/**
 * The interface represents a filter
 * needed for implementing a database query.
 * 
 * @author Ivan Rep
 */
public interface IFilter {

	/**
	 * The method must check if the student record
	 * will be accepted.
	 * 
	 * @param record the given student record
	 * @return true if it is accepted, otherwise false
	 */
	public boolean accepts(StudentRecord record);

}
