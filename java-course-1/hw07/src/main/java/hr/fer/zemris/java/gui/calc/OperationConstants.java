package hr.fer.zemris.java.gui.calc;

import static java.lang.Math.acos;
import static java.lang.Math.asin;
import static java.lang.Math.cos;
import static java.lang.Math.log;
import static java.lang.Math.log10;
import static java.lang.Math.pow;
import static java.lang.Math.sin;
import static java.lang.Math.tan;
import static java.lang.Math.atan;

/**
 * Class contains all used IOperations defined as constants.
 * 
 * @author Ivan Rep
 */
public class OperationConstants {

	/**
	 * Defines the operation 1/x. Its' inverse is x^1.
	 */
	public static final IOperation ONE_OVER_X = new IOperation() {
		@Override
		public double DoOperation(double operand) { return 1.0/operand; }
		@Override
		public double DoInverseOperation(double operand) { return operand; }
	};

	/**
	 * Defines the operation sin(x). Its' inverse is arcsin(x).
	 */
	public static final IOperation SINE = new IOperation() {
		@Override
		public double DoOperation(double operand) { return sin(operand); }
		@Override
		public double DoInverseOperation(double operand) { return asin(operand); }
	};

	/**
	 * Defines the operation logarithm of base 10. Its' inverse is 10^n.
	 */
	public static final IOperation LOG_BASE_10 =  new IOperation() {
		@Override
		public double DoOperation(double operand) { return log10(operand); }
		@Override
		public double DoInverseOperation(double operand) { return pow(10, operand); }
	};

	/**
	 * Defines the operation cos(x). Its' inverse is arccos(x).
	 */
	public static final IOperation COSINE =	new IOperation() {
		@Override
		public double DoOperation(double operand) { return cos(operand); }
		@Override
		public double DoInverseOperation(double operand) { return acos(operand); }
	};

	/**
	 * Defines the operation logarithm of base e. Its' inverse is e^n.
	 */
	public static final IOperation LOG_BASE_E = new IOperation() {
		@Override
		public double DoOperation(double operand) { return log(operand); }
		@Override
		public double DoInverseOperation(double operand) { return pow(Math.E, operand); }
	};
	
	/**
	 * Defines the operation tan(x). Its' inverse is arctan(x).
	 */
	public static final IOperation TAN = new IOperation() {
		@Override
		public double DoOperation(double operand) { return tan(operand); }
		@Override
		public double DoInverseOperation(double operand) { return atan(operand); }
	};
	
	/**
	 * Defines the operation cotan(x). Its' inverse is arcctg(x).
	 */
	public static final IOperation COTAN = new IOperation() {
		@Override
		public double DoOperation(double operand) { return 1/tan(operand); }
		@Override
		public double DoInverseOperation(double operand) { return 1/atan(operand); }
	};

}
