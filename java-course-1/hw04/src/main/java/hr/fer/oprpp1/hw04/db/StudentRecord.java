package hr.fer.oprpp1.hw04.db;

/**
 * The class represents a student record containing the jmbag, name, surname and final grade.
 * 
 * @author Ivan Rep
 */
public class StudentRecord {

	private String jmbag;
	private String lastName;
	private String firstName;
	private int finalGrade;

	/**
	 * The constructor that assigns all the values.
	 * 
	 * @param jmbag the assigned jmbag value
	 * @param lastName the assigned lastName value
	 * @param firstName the assigned firstName value
	 * @param finalGrade the assigned finalGrade value
	 */
	public StudentRecord(String jmbag, String lastName, String firstName, int finalGrade) {
		this.jmbag = jmbag;
		this.lastName = lastName;
		this.firstName = firstName;
		this.finalGrade = finalGrade;
	}

	/**
	 * The getter method for the jmbag attribute.
	 * 
	 * @return the jmbag of the student record.
	 */
	public String getJmbag() {
		return jmbag;
	}
	
	/**
	 * The getter method for the last name attribute.
	 * 
	 * @return the last name of the student record.
	 */
	public String getLastName() {
		return lastName;
	}

	/**
	 * The getter method for the first name attribute.
	 * 
	 * @return the first name of the student record.
	 */
	public String getFirstName() {
		return firstName;
	}

	/**
	 * The getter method for the final grade attribute.
	 * 
	 * @return the final grade of the student record.
	 */
	public int getFinalGrade() {
		return finalGrade;
	}

	/**
	 * The method returns the hash code for the student
	 * record calculated based on jmbag.
	 * 
	 * @return the hash code for the student record
	 */
	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + ((jmbag == null) ? 0 : jmbag.hashCode());
		return result;
	}

	/**
	 * The method checks if the given object's and the current 
	 * student record's jmbags are equal.
	 * 
	 * @param obj the given object
	 * @return true if jmbags are equal, otherwise false
	 */
	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		StudentRecord other = (StudentRecord) obj;
		if (jmbag == null) {
			if (other.jmbag != null)
				return false;
		} else if (!jmbag.equals(other.jmbag))
			return false;
		return true;
	}


}
