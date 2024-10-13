package hr.fer.oprpp1.custom.collections.demo;

import hr.fer.oprpp1.custom.collections.*;

/**
 * The class demonstrates the functionalities of the Tester class.
 * 
 * @author Ivan Rep
 */
class EvenIntegerTester implements Tester {

	/**
	 * The method executes once the program starts.
	 * 
	 * @param args the arguments of the command line
	 */
	public static void main(String[] args) {

		Tester t = new EvenIntegerTester();
		System.out.println(t.test("Ivo"));
		System.out.println(t.test(22));
		System.out.println(t.test(3));

	}

	/**
	 * The method is used for testing whether the objects are even integers.
	 * 
	 * @return true if the element is an even Integer object, otherwise false
	 */
	public boolean test(Object obj) {
		if(!(obj instanceof Integer)) return false;
		Integer i = (Integer)obj;
		return i % 2 == 0;
	}

}
