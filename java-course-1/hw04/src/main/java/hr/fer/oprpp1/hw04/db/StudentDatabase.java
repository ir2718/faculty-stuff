package hr.fer.oprpp1.hw04.db;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.stream.Collectors;

/**
 * The class represents a small database of students.
 * 
 * @author Ivan Rep
 */
public class StudentDatabase {
	
	private HashMap<String, StudentRecord> mapOfStudents;
	private ArrayList<StudentRecord> listOfStudents;

	/**
	 * The constructor method for the student database.
	 * It takes in a list with lines in the format
	 * "jmbag lastName(s) firstName finalGrade".
	 * 
	 * @param list the list of data about students
	 */
	public StudentDatabase( List<String> list ) {
		mapOfStudents = new HashMap<>();
		listOfStudents = new ArrayList<>();
		
		String[] arr = null;
		for( String str : list ) {
			arr = str.split(" ");
			
			StudentRecord rec = null;
			if (arr.length==4)	
				rec = new StudentRecord( arr[0], arr[1], arr[2], Integer.parseInt(arr[3]) );	// jmbag surname name grade
			else if (arr.length==5)
				rec = new StudentRecord( arr[0], arr[1]+" "+arr[2], arr[3], Integer.parseInt(arr[4]) ); // jmbag surname1 surname2 name grade
			
			listOfStudents.add(rec);
			mapOfStudents.put( arr[0], rec);
			
		}
			
	}
	
	/**
	 * The method returns the student record for the given jmbag.
	 * 
	 * @param jmbag the given jmbag
	 * @return the student record for the given jmbag
	 */
	public StudentRecord forJMBAG(String jmbag) {
		return mapOfStudents.get(jmbag);
	}
	
	/**
	 * The method returns a list of student records
	 * that are filtered by the given filter.
	 * 
	 * @param filter the given filter
	 * @return the filteder list of students
	 */
	public List<StudentRecord> filter(IFilter filter){
		return listOfStudents
				.stream()
				.filter( record -> filter.accepts(record) )
				.collect( Collectors.toList() );
	}
	
	
}
