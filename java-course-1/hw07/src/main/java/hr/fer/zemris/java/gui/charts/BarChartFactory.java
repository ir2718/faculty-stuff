package hr.fer.zemris.java.gui.charts;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner;

/**
 * Class represents factory for bar chart.
 * Creates a BarChart object from the given file.
 * 
 * @author Ivan Rep
 */
public class BarChartFactory {
	
	public static final int ROW_CONSTANT = 6;
	
	public static final int ZERO = 0;
	public static final int ONE = 1;
	
	/**
	 * Main method. Parses data from the file and uses it to create a BarChart object.
	 * 
	 * @param file			file that is read
	 * @return				new BarChart object created from the file
	 * @throws IOException	if the given file is not .txt format
	 */
	public static BarChart parse(File file) throws IOException {
		
		String s = file.getName().substring( file.getName().lastIndexOf('.'), file.getName().length() );
		
		if( !s.toLowerCase().equals(".txt") ) throw new IOException("The given file is in the wrong format.");
		
		String descX = "";
		String descY = "";
		ArrayList<XYValue> list = new ArrayList<>();
		int yMin = -1;
		int yMax = -1;
		int diffY = -1;
		
		try {

			Scanner sc = new Scanner(file);
			
			for(int i=0; i<ROW_CONSTANT; i++)
				switch(i) {
				case 0 -> descX = sc.nextLine().trim();
				case 1 -> descY = sc.nextLine().trim();
				case 2 -> list = MakeList(sc.nextLine().trim());
				case 3 -> yMin = Integer.parseInt(sc.nextLine());
				case 4 -> yMax = Integer.parseInt(sc.nextLine());
				case 5 -> diffY = FindDIffY( Integer.parseInt(sc.nextLine()), yMax);
				}
			sc.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch(Exception exc) {
			throw new BarChartParserException("There's a problem with the given file. Try again.");
		}
		
		return new BarChart( list, descX, descY, yMin, yMax, diffY );
	}

	/**
	 * Finds the difference between two y values.
	 * 
	 * @param parseInt					value for the difference read from the file
	 * @param max						the given maximum value
	 * @return							new difference between two y values
	 * @throws BarChartParserException	if there is no number in range divisible by the maximum
	 */
	private static int FindDIffY(int parseInt, int max) {
		for(int i=parseInt; i<max; i++ ) if(max%i==0) return i;
		throw new BarChartParserException();
	}

	/**
	 * Creates a list of XYValue objects for the bar chart.
	 * 
	 * @param str	the given string containing the x and y values of the points
	 * @return		an array list of XYValue objects for the bar chart
	 */
	private static ArrayList<XYValue> MakeList(String str) {

		ArrayList<XYValue> list = new ArrayList<>();
		Arrays.stream( str.split(" ") )
			.forEach( s -> {
				String[] temp = s.split(",");
				list.add( new XYValue( Integer.parseInt( temp[ZERO] ), Integer.parseInt( temp[ONE] ) ) ); } );
		
		return list;
	}
	
	/**
	 * Class represents an exception used in parsing the bar chart from the file
	 * 
	 * @author Ivan Rep
	 */
	private static class BarChartParserException extends RuntimeException {
		
		/**
		 * Constructor. Sets the message.
		 * 
		 * @param message the given message
		 */
		public BarChartParserException(String message) { super(message); }
		
		/**
		 * Empty constructor.
		 */
		public BarChartParserException() { super(); }
	}
	
}
