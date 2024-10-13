package hr.fer.zemris.java.gui.calc;

/**
 * Interface used as a strategy for unary operations.
 * 
 * @author Ivan Reo
 */
public interface IOperation {

	/**
	 * Method finds and executes the operation or its inverse.
	 * 
	 * @param doInverse if true executes the inverse function,
	 * 					otherwise executes the current function
	 * @param model 	reference to the model of the used calculator
	 */
	default void FindAndDoOperation(boolean doInverse, CalcModelImpl model) {
		double result = model.getValue();
		double val = doInverse ? DoInverseOperation(result) : DoOperation(result);
		model.setValue( val );
		model.clearActiveOperand();
	}
	
	/**
	 * Method should define an operation and calculate the
	 * value with the given operand.
	 * 
	 * @param operand 	the given operand
	 * @return			result of the function 
	 */
	abstract double DoOperation(double operand);
	
	/**
	 * Method should define an inverse operation and calculate the
	 * value with the given operand.
	 * 
	 * @param operand 	the given operand
	 * @return			result of the inverse function 
	 */
	abstract double DoInverseOperation(double operand);
	
}
