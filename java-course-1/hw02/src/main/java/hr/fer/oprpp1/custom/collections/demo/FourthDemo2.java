package hr.fer.oprpp1.custom.collections.demo;

import hr.fer.oprpp1.custom.collections.*;

/**
 * The class demonstrates the functionalities of the collection classes.
 * 
 * @author Ivan Rep
 */
public class FourthDemo2 {

	/**
	 * The method executes once the program starts.
	 * 
	 * @param args the arguments of the command line
	 */
	public static void main(String[] args) {
		Collection col = new LinkedListIndexedCollection(); // new ArrayIndexedCollection(); 
		col.add("Ivo");
		col.add("Ana");
		col.add("Jasna");
		ElementsGetter getter1 = col.createElementsGetter();
		ElementsGetter getter2 = col.createElementsGetter();
		System.out.println("Jedan element: " + getter1.getNextElement());
		System.out.println("Jedan element: " + getter1.getNextElement());
		System.out.println("Jedan element: " + getter2.getNextElement());
		System.out.println("Jedan element: " + getter1.getNextElement());
		System.out.println("Jedan element: " + getter2.getNextElement());
		}


}
