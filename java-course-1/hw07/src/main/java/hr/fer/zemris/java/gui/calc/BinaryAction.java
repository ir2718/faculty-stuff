package hr.fer.zemris.java.gui.calc;

import java.util.function.DoubleBinaryOperator;

import hr.fer.zemris.java.gui.calc.model.CalculatorInputException;

/**
 * Interface is used as a strategy for binary operations.
 * 
 * @author Ivan Rep
 */

public interface BinaryAction {
	
	/**
	 * Method does the default job for each binary operation. 
	 * Applies the operation and refreshes the display.
	 * 
	 * @param model model of a calculator that is being used
	 * @param op 	operation that needs to be done
	 */
	default void defaultJob(CalcModelImpl model, DoubleBinaryOperator op) {
		
		if( model.isActiveOperandSet() ) {
			double result = model.getPendingBinaryOperation().applyAsDouble(model.getActiveOperand(), model.getValue());
//			model.setActiveOperand(result);
			model.setFreezeValue(model.toString());
			model.setValue(result);
//			if( model.getPendingBinaryOperation()!=null && !model.getPendingBinaryOperation().equals(op) ) model.setPendingBinaryOperation(op); 
//		} 
//		else {
//			model.setActiveOperand(model.getValue());
//			model.setPendingBinaryOperation( op );
		}
		model.setActiveOperand(model.getValue());
		model.setPendingBinaryOperation(op);
	}
	
}
