package hr.fer.oprpp1.hw04.db;

/**
 * The class represents the use of the IFieldValueGetter
 * interface by implementing getters for the firstName, lastName and jmbag.
 * 
 * @author Ivan Rep
 */
public class FieldValueGetters {

	/**
	 * Returns the firstName attribute from the given student record.
	 * 
	 * @param record the given student record
	 * @return the first name from the given student record
	 */
	public static final IFieldValueGetter FIRST_NAME = record -> record.getFirstName();
	
	/**
	 * Returns the lastName attribute from the given student record.
	 * 
	 * @param record the given student record
	 * @return the last name from the given student record
	 */
	public static final IFieldValueGetter LAST_NAME = record -> record.getLastName();
	
	/**
	 * Returns the jmbag attribute from the given student record.
	 * 
	 * @param record the given student record
	 * @return the jmbag from the given student record
	 */
	public static final IFieldValueGetter JMBAG = record -> record.getJmbag();

}
