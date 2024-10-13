package hr.fer.zemris.java.gui.calc;

import javax.swing.JButton;

/**
 * Class represents a digit button used for the calculator.
 * 
 * @author Ivan Rep
 */
public class DigitButton extends JButton {

	private static final long serialVersionUID = 1L;
	
	private String digit;
	private JButton button;
	
	/**
	 * Constructor method.
	 * 
	 * @param digit the given digit
	 */
	public DigitButton(String digit) {
		this.digit = digit;
		this.button = new JButton(digit);
		this.button.setFont(this.button.getFont().deriveFont(30f));
	}
	
	/**
	 * Getter method for the digit.
	 * 
	 * @return the string representation of the digit
	 */
	public String getDigit() {
		return this.digit;
	}
	
	/**
	 * Getter method for the digit button.
	 * 
	 * @return a JButton object reference of the digit button
	 */
	public JButton getButton() {
		return this.button;
	}
}
