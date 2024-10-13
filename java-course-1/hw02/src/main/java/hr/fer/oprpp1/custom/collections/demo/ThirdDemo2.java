package hr.fer.oprpp1.custom.collections.demo;

import hr.fer.oprpp1.custom.collections.*;

/**
 * The class demonstrates the functionalities of the collection classes.
 * 
 * @author Ivan Rep
 */
public class ThirdDemo2 {

	/**
	 * The method executes once the program starts.
	 * 
	 * @param args the arguments of the command line
	 */
	public static void main(String[] args) {
		Collection col =  new LinkedListIndexedCollection(); // new ArrayIndexedCollection(); 
		col.add("Ivo");
		col.add("Ana");
		col.add("Jasna");
		ElementsGetter getter = col.createElementsGetter();
		System.out.println("Jedan element: " + getter.getNextElement());
		System.out.println("Jedan element: " + getter.getNextElement());
		System.out.println("Jedan element: " + getter.getNextElement());
		System.out.println("Jedan element: " + getter.getNextElement());
		}


}
