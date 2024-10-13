package hr.fer.oprpp1.hw04.db;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Optional;
import java.util.Scanner;

/**
 * The class demostrates the functionalities of the database queries.
 * 
 * @author Ivan Rep
 *
 */
public class StudentDB {

	public static final int INDEX_AFTER_QUERY = 5;
	public static final int JMBAG_LENGTH = 10;
	public static final int FINAL_GRADE_LENGTH = 1;
	
	/**
	 * The main method that executes once the program starts.
	 * 
	 * @param args arguments of the command line, not used here
	 * @throws IOException if there is no database.txt file in the 
	 * current directory
	 */
	public static void main(String[] args) throws IOException {
		
		List<String> lines = Files.readAllLines(
				 Paths.get("./src/main/resources/database.txt"),
				 StandardCharsets.UTF_8
				);

		List<String> jmbags = new ArrayList<>();
		List<String> linesProcessed = new ArrayList<>();
		
		for ( String line : lines ) {
			String s = line;
			String[] strArr = s.replaceAll("\\s+", " ").split(" ");
			
			if( jmbags.contains(strArr[0]) ) 
				throw new IllegalArgumentException("There are two students with the same jmbag("+strArr[0]+") in database.txt.");
			if( Integer.parseInt(strArr[strArr.length-1])<1 || Integer.parseInt(strArr[strArr.length-1])>5 ) 
				throw new IllegalArgumentException("The student with jmbag "+strArr[0]+" has a grade greater than 5 or less than 1.");
			
			linesProcessed.add(s.replaceAll("\\s+", " "));
			jmbags.add(strArr[0]);
		}

		StudentDatabase db = new StudentDatabase(linesProcessed);
		
		DatabaseStartupPrint();
		Scanner sc2 = new Scanner(System.in);
		
		QueryParser parser = null;
		ArrayList<StudentRecord> printList = null;
		
		
		while( sc2.hasNextLine() ) {
			String line = sc2.nextLine();
			
			if( line.equals("exit") ) {
				System.out.println("Goodbye!");
				break;
			}

			if(!line.trim().startsWith("query")) {
				sc2.close();
				throw new IllegalArgumentException("Every query must start with the key word query.");
			}

			String query = line.trim().substring(INDEX_AFTER_QUERY);
			parser = new QueryParser(query);
			
			if(parser.isDirectQuery()) {
				StudentRecord r = db.forJMBAG(parser.getQueriedJMBAG());
				DirectOutput(r);
			} else {
				printList = new ArrayList<>();
				for(StudentRecord r : db.filter(new QueryFilter(parser.getQuery())))
					printList.add(r);
				IndirectOutput(printList);
			}
			
			System.out.print("> ");
		}
		
		sc2.close();

	}

	/**
	 * The method used for printing the output of direct queries.
	 * 
	 * @param r the student record that is printed
	 */
	private static void DirectOutput(StudentRecord r) {
		if(r==null)
			System.out.println("Records selected: 0\n");
		else {
			int len = 1;
			int maxLast = r.getLastName().length();
			int maxFirst = r.getFirstName().length();
			System.out.println("Using index for record retrieval.");
			printTable( maxLast, maxFirst );
			printStudentRecord( r, maxLast, maxFirst );
			printTable( maxLast, maxFirst );
			System.out.println("Records selected: "+len+"\n");
		}
	}
	
	/**
	 * The method used for printing the student record.
	 * 
	 * @param r the student record that are printed
	 * @param maxLast the maximum length of a last name in a query result 
	 * @param maxFirst the maximum length of a first name in a query result
	 */
	private static void printStudentRecord(StudentRecord r, int maxLast, int maxFirst) {
		String lastName = r.getLastName();
		String firstName = r.getFirstName();
		
		StringBuilder sb = new StringBuilder("| ");
		sb.append(r.getJmbag()).append(" | ");
		sb.append(lastName).append(" ".repeat(maxLast-lastName.length())).append(" | ");
		sb.append(firstName).append(" ".repeat(maxFirst-firstName.length())).append(" | ");
		sb.append(r.getFinalGrade()).append(" |");
		System.out.println(sb.toString());
	}

	/**
	 * The method used for printing the output of indirect queries.
	 * 
	 * @param r the list of student records that are printed
	 */
	private static void IndirectOutput(ArrayList<StudentRecord> printList) {
		if( printList.isEmpty() )
			System.out.println("Records selected: 0\n");
		else {
			int len = printList.size();
			Optional<Integer> maxLast  = printList.stream().map( rec -> rec.getLastName().length() ).max(Comparator.naturalOrder());
			Optional<Integer> maxFirst = printList.stream().map( rec -> rec.getFirstName().length() ).max(Comparator.naturalOrder());
			printTable( maxLast.get(), maxFirst.get() );
			for( StudentRecord rec : printList )
				printStudentRecord( rec, maxLast.get(), maxFirst.get() );
			printTable( maxLast.get(), maxFirst.get() );
			System.out.println("Records selected: "+len+"\n");
		}
	}

	/**
	 * Prints the database startup message.
	 */
	private static void DatabaseStartupPrint() {
		System.out.println("Type a query for the database:");
		System.out.print(">");
	}


	/**
	 * Prints the borders in the table.
	 * 
	 * @param maxLast the maximum length of a last name in a query result 
	 * @param maxFirst the maximum length of a first name in a query result
	 */
	private static void printTable(int maxLast, int maxFirst) {
		System.out.print("+" + "=".repeat(JMBAG_LENGTH+2) + "+" );
		System.out.print( "=".repeat(maxLast+2) + "+" );
		System.out.print( "=".repeat(maxFirst+2) + "+" );
		System.out.print( "=".repeat(FINAL_GRADE_LENGTH+2) + "+\n" );
	}
	
}
