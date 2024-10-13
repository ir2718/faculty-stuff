package hr.fer.oprpp1.custom.collections.demo;

import hr.fer.oprpp1.custom.collections.*;

/**
 * The class demonstrates the functionalities of the collection classes.
 * 
 * @author Ivan Rep
 */
public class FifthDemo2 {

	/**
	 * The method executes once the program starts.
	 * 
	 * @param args the arguments of the command line
	 */
	public static void main(String[] args) {
		Collection col1 = new LinkedListIndexedCollection(); // new ArrayIndexedCollection();
		Collection col2 = new LinkedListIndexedCollection(); // new ArrayIndexedCollection();
		col1.add("Ivo");
		col1.add("Ana");
		col1.add("Jasna");
		col2.add("Jasmina");
		col2.add("Štefanija");
		col2.add("Karmela");
		ElementsGetter getter1 = col1.createElementsGetter();
		ElementsGetter getter2 = col1.createElementsGetter();
		ElementsGetter getter3 = col2.createElementsGetter();
		System.out.println("Jedan element: " + getter1.getNextElement());
		System.out.println("Jedan element: " + getter1.getNextElement());
		System.out.println("Jedan element: " + getter2.getNextElement());
		System.out.println("Jedan element: " + getter3.getNextElement());
		System.out.println("Jedan element: " + getter3.getNextElement());
		}


}
