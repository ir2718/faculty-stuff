package hr.fer.oprpp1.hw04.db;

/**
 * The interface represents a field getter 
 * needed for implementing a database query.
 * 
 * @author Ivan Rep
 */
public interface IFieldValueGetter {

	/**
	 * The method must return an attribute of 
	 * the student record.
	 * 
	 * @param record the given student record
	 * @return an attribute of the student record
	 */
	public String get(StudentRecord record);
	
}
