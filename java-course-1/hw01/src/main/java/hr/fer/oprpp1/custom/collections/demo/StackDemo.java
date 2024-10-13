package hr.fer.oprpp1.custom.collections.demo;

import hr.fer.oprpp1.custom.collections.ObjectStack;

/**
 * This class is used for the demonstration of the class ObjectStack.
 * 
 * @author Ivan Rep
 */
public class StackDemo {

	/**
	 * The method called once the program starts.
	 * 
	 * @param args the arguments of the command line
	 * @throws Exception if the given input can't be parsed
	 */
	public static void main(String[] args) throws Exception {

		ObjectStack s = new ObjectStack();
		String[] arr = args[0].split(" ");

		for(String str : arr) {

			if(isInteger(str)) 
				s.push(Integer.parseInt(str));
			else if(isOperator(str))
				s.push( PerformOperation( (Integer)s.pop(), (Integer)s.pop(), str ) );

		}

		if (s.size()!=1) throw new Exception("The entered input is not compatible!");

		System.out.println(s.pop());

	}

	/**
	 * The method performs an operation on the two parameters depending on the given operator.
	 * 
	 * @param num the second parameter of the operation
	 * @param num2 the first parameter of the operation
	 * @param op the String representing the operation
	 * @return value that is computed from the parameters and the operation sign
	 * @throws ArithmeticException if the second parameter is 0 and the operator is division
	 */
	private static int PerformOperation(Integer num, Integer num2, String op) {
		if(op.equals("/") && num==0) throw new ArithmeticException("You can't divide a number by 0!");

		return switch (op) {
		case "+" -> num2 + num;
		case "-" -> num2 - num;
		case "/" -> num2 / num;
		case "*" -> num2 * num;
		case "%" -> num2 % num;
		default -> throw new IllegalArgumentException("Unexpected value: "+op+". The supported values are +, -, *, // and %.");
		};

	}

	/**
	 * The method checks whether a String is an operator.
	 * 
	 * @param str the String to be checked
	 * @return true if it is an operator, otherwise false
	 */
	private static boolean isOperator(String str) {
		String operators = "+-/%*";
		return operators.contains(str) && str.length()==1;
	}
	
	/**
	 * The method checks whether a String is an Integer.
	 * 
	 * @param str the String to be checked
	 * @return true if it is an Integer, otherwise false
	 */
	private static boolean isInteger(String str) {
		char[] arr = str.toCharArray();

		for(int i=0, m=str.length(); i<m; i++)
			if(Character.isDigit(arr[i]) || (i==0 && arr[i]=='-') );
			else return false;

		return true;
	}


}
