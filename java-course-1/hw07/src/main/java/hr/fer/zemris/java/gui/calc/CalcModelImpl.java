package hr.fer.zemris.java.gui.calc;

import java.util.ArrayList;
import java.util.Stack;
import java.util.function.DoubleBinaryOperator;

import hr.fer.zemris.java.gui.calc.model.CalcModel;
import hr.fer.zemris.java.gui.calc.model.CalcValueListener;
import hr.fer.zemris.java.gui.calc.model.CalculatorInputException;
import static java.lang.Math.*;

/**
 * Class used for implementing the logic behind a calculator.
 * 
 * @author Ivan Rep
 */
public class CalcModelImpl implements CalcModel {

	private boolean editable;
	private boolean sign;
	private String enteredSign;
	private Double value;
	private String frozenValue;

	private Double activeOperand;
	private DoubleBinaryOperator pendingOperation;
	
	private ArrayList<CalcValueListener> listeners;
	
	/**
	 * Constructor. Sets the values of the variables.
	 */
	public CalcModelImpl() {
		this.editable = true;
		this.sign = true;
		this.enteredSign = "";
		this.value = 0.0;
		this.listeners = new ArrayList<>();
	}
	
	/**
	 * Notifies all the registered listeners.
	 */
	private void NotifyListeners() {
		listeners.stream().forEach( l -> l.valueChanged(this) );
	}
	
	/**
	 * Adds the given listener to the list of registered listeners.
	 * 
	 * @param CalcValueListener l the given listener
	 */
	@Override
	public void addCalcValueListener(CalcValueListener l) {
		listeners.add(l);
	}

	/**
	 * Removes the given listener from the list of registered listeners.
	 * 
	 * @param CalcValueListener l the given listener
	 */
	@Override
	public void removeCalcValueListener(CalcValueListener l) {
		listeners.remove(l);
	}

	/**
	 * Returns the current value.
	 * 
	 * @return the current value
	 */
	@Override
	public double getValue() {
		return value;
	}
	
	/**
	 * Sets the value parameter to the given value.
	 * 
	 * @param value the given value
	 */
	@Override
	public void setValue(double value) {
		this.value = value;
		this.enteredSign = String.valueOf(value);
		editable = false;
		NotifyListeners();
	}

	/**
	 * Returns if the calculator is editable.
	 * 
	 * @return true if editable, otherwise false
	 */
	@Override
	public boolean isEditable() {
		return this.editable;
	}

	/**
	 * Clears the value, entered sign and editable parameters.
	 */
	@Override
	public void clear() {
		this.value = 0.0;
		this.enteredSign = "";
		this.editable=true;
		this.NotifyListeners();
	}

	/**
	 * Clears all the parameters.
	 */
	@Override
	public void clearAll() {
		this.activeOperand = null;
		this.pendingOperation = null;
		clear();
	}

	/**
	 * Swaps the sign of the current value.
	 * 
	 * @throws CalculatorInputException if the calculator is not editable
	 */
	@Override
	public void swapSign() throws CalculatorInputException {
		if( !this.editable ) throw new CalculatorInputException("The calculator is currently not editable.");

		if(!this.enteredSign.isEmpty())
			this.enteredSign = this.sign ? "-"+this.enteredSign : this.enteredSign.substring(1) ;

		this.sign = !sign;
		
		if(!this.enteredSign.isEmpty())
		this.value = Double.valueOf(this.enteredSign);
		
		this.frozenValue = null;
		this.NotifyListeners();
	}

	/**
	 * Inserts a decimal point to the current value.
	 * 
	 * @throws CalculatorInputException if the calculator isn't editable, 
	 * if there is already a decimal point in the number or if there are no digits in the number
	 */
	@Override
	public void insertDecimalPoint() throws CalculatorInputException {
		if( !this.editable ) throw new CalculatorInputException("The calculator is currently not editable.");
		if( this.enteredSign.contains(".") ) throw new CalculatorInputException("There is already a decimal point in the number.");
		if( this.enteredSign.isEmpty() ) throw new CalculatorInputException("The number must start with a digit.");
		this.enteredSign += ".";
		this.frozenValue = null;
		this.NotifyListeners();
	}

	/**
	 * Appends the given digit to the current value.
	 * 
	 * @param digit						the given digit
	 * @throws CalculatorInputException if the input isn't editable or if the number is infinite
	 * @throws IllegalArgumentException if the digit isn't parseable
	 */
	@Override
	public void insertDigit(int digit) throws CalculatorInputException, IllegalArgumentException {
		if( !this.isEditable() ) throw new CalculatorInputException("The input isn't editable.");

		if( this.hasFrozenValue() ) {
			enteredSign = "";
			frozenValue = "";
		}
		
		String curr = this.enteredSign + String.valueOf(digit);
		Double i = null;

		try { i = Double.parseDouble(curr); }
		catch ( NullPointerException  | NumberFormatException exc ) { throw new CalculatorInputException("The digit isn't parseable."); }

		if(i.isInfinite()) throw new CalculatorInputException("The number is too big.");

		this.enteredSign += String.valueOf(digit);
		this.value = Double.parseDouble(this.enteredSign);
		this.frozenValue=null;
		this.NotifyListeners();
	}

	/**
	 * Checks if the active operand is set.
	 * 
	 * @return true if the active operand is set, otherwise false
	 */
	@Override
	public boolean isActiveOperandSet() {
		if(this.activeOperand==null) return false;
		return true;
	}

	/**
	 * Getter method for the active operand.
	 * 
	 * @throws IllegalStateException if the active operand is not set
	 */
	@Override
	public double getActiveOperand() throws IllegalStateException {
		if(this.activeOperand==null) throw new IllegalStateException("The active operand is not set.");
		return this.activeOperand;
	}

	/**
	 * Setter method for the activeOperand parameter.
	 * 
	 * @param activeOperand the given activeOperand
	 */
	@Override
	public void setActiveOperand(double activeOperand) {
		this.activeOperand = activeOperand;
		this.frozenValue = String.valueOf( getValue() );
		this.editable = true;
	}

	/**
	 * Clears the active operand.
	 */
	@Override
	public void clearActiveOperand() {
		this.activeOperand=null;
		this.setFreezeValue(value.toString());
		this.editable=true;
	}

	/**
	 * Getter method for the pending binary operation.
	 * 
	 * @return the pending operation parameter
	 */
	@Override
	public DoubleBinaryOperator getPendingBinaryOperation() {
		return this.pendingOperation;
	}

	/**
	 * Setter method for the double binary operator.
	 * 
	 * @param op the given double binary operator
	 */
	@Override
	public void setPendingBinaryOperation(DoubleBinaryOperator op) {
		pendingOperation = op;
	}

	/**
	 * Returns a string representation of the current value of the calculator.
	 * 
	 * @return the string representation of the calculator model implementation
	 */
	@Override
	public String toString() {
		if(this.value==null || abs(this.value)==0 ) return sign ? "0" : "-0";

		Double d = Double.parseDouble(this.enteredSign);

		return ( !this.enteredSign.contains(".") && Double.isFinite(d)  && !Double.isNaN(d) ) ? 
				Integer.toString( value.intValue() ) :
					Double.toString(value);
	}

	/**
	 * Setter method for the frozen value paramterer.
	 * 
	 * @param frozenValue the given frozen value
	 */
	public void setFreezeValue(String frozenValue) {
		this.frozenValue = frozenValue;
	}

	/**
	 * Checks whether the current frozen value is not null.
	 * 
	 * @return true if not null, otherwise false
	 */
	public boolean hasFrozenValue() {
		return this.frozenValue!=null;
	}


}
