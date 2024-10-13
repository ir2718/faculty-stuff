package hr.fer.oprpp1.hw01.demo;

import hr.fer.oprpp1.hw01.ComplexNumber;

/**
 * This class is used for demonstration of the ComplexNumber class functionality
 * 
 * @author Ivan Rep
 */
public class ComplexDemo {

	/**
	 * The method is called once the program starts.
	 * 
	 * @param args command line arguments, they're not used
	 */
	public static void main(String[] args) {
		
		ComplexNumber c1 = new ComplexNumber(2, 3);
		ComplexNumber c2 = ComplexNumber.parse("2.5-3i");
		ComplexNumber c3 = c1.add(ComplexNumber.fromMagnitudeAndAngle(2, 1.57))
		.div(c2).power(3).root(2)[1];
		System.out.println(c3);
		
	}

}
