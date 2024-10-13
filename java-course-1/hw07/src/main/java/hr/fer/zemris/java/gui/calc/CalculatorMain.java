package hr.fer.zemris.java.gui.calc;

import java.awt.Color;
import static java.lang.Math.*;
import java.awt.Container;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.ArrayList;
import java.util.Stack;
import java.util.function.DoubleBinaryOperator;

import javax.swing.BorderFactory;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.SwingUtilities;
import javax.swing.WindowConstants;

import hr.fer.zemris.java.gui.calc.model.CalcModel;
import hr.fer.zemris.java.gui.layouts.CalcLayout;

/**
 * Class demonstrates the functionalities of the calculator.
 * 
 * @author Ivan Rep
 */
public class CalculatorMain extends JFrame {

	private static final long serialVersionUID = 1L;

	private CalcModelImpl model;
	private JCheckBox inv;
	private ArrayList<ActionButton> arrOfActionButtons;
	private Stack<Double> stack;

	/**
	 * Constructor. Initializes the frame and initializes the needed variables.
	 */
	public CalculatorMain() {
		super();
		setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);
		setTitle("Java Calculator v1.0");
		setLocation(20, 20);
		setSize(700, 500);

		JPanel p = new JPanel();
		arrOfActionButtons = new ArrayList<>();
		stack = new Stack<>();
//		p.setBorder(BorderFactory.createLineBorder(Color.BLUE, 4));
		setContentPane(p);

		model = new CalcModelImpl();

		initGUI();
	}

	/**
	 * Method creates the calculator buttons and labels and 
	 * connects the functionalities with the GUI.
	 */
	private void initGUI() {
		Container cp = getContentPane();
		cp.setLayout(new CalcLayout(5));


		// 1st row
		JLabel display = new JLabel("0");
		display.setFont(display.getFont().deriveFont(30f));
		display.setHorizontalAlignment(JLabel.RIGHT);
		display.setOpaque(true);
		display.setBackground(Color.YELLOW);
		
		model.addCalcValueListener( l -> display.setText(model.toString()) );
		cp.add( display, "1, 1" );
		cp.add( createNonNumberButton("=",
				(e) -> { 
					double result = model.getPendingBinaryOperation().applyAsDouble(model.getActiveOperand(), model.getValue());
					model.setValue(result);
					model.clearActiveOperand();
					model.setPendingBinaryOperation(null);
					} ), "1, 6" );
		cp.add( createNonNumberButton("clr", e -> this.model.clear() ) ,  "1, 7");

		// 2nd row
		cp.add( createUnaryActionButton("1/x", "x^1", e -> OperationConstants.ONE_OVER_X.FindAndDoOperation(this.inv.isSelected(), model) ), "2, 1" );
		cp.add( createUnaryActionButton("sin", "arcsin", e -> OperationConstants.SINE.FindAndDoOperation(this.inv.isSelected(), model) ), "2, 2" );
		cp.add( createNumberButton("7", e -> this.model.insertDigit(7) )  ,"2, 3");
		cp.add( createNumberButton("8", e -> this.model.insertDigit(8) )  ,"2, 4");
		cp.add( createNumberButton("9", e -> this.model.insertDigit(9) )  ,"2, 5");
		cp.add( createNonNumberButton("/", e -> {
			BinaryAction action = new BinaryAction() {};
			action.defaultJob(model, (a, b) -> a/b ); 
		}) ,"2, 6");
		cp.add( createNonNumberButton("reset", e -> model.clearAll() )  ,"2, 7");

		// 3rd row
		cp.add( createUnaryActionButton("log", "10^x", e -> OperationConstants.LOG_BASE_10.FindAndDoOperation(this.inv.isSelected(), model) ), "3, 1" );
		cp.add( createUnaryActionButton("cos", "arccos", e -> OperationConstants.COSINE.FindAndDoOperation(this.inv.isSelected(), model) ), "3, 2" );
		cp.add( createNumberButton("4", e -> this.model.insertDigit(4) )  ,"3, 3");
		cp.add( createNumberButton("5", e -> this.model.insertDigit(5) )  ,"3, 4");
		cp.add( createNumberButton("6", e -> this.model.insertDigit(6) )  ,"3, 5");
		cp.add( createNonNumberButton("*", e -> { 
			BinaryAction action = new BinaryAction() {};
			action.defaultJob(model, (a, b) -> a*b );
		}) ,"3, 6");
		cp.add( createNonNumberButton("push", e -> {
			stack.push(model.getValue());
		} )  ,"3, 7");

		// 4th row
		cp.add( createUnaryActionButton("ln", "e^x",  e -> OperationConstants.LOG_BASE_E.FindAndDoOperation(this.inv.isSelected(), model) ), "4, 1" );
		cp.add( createUnaryActionButton("tan", "arctan", e ->  OperationConstants.TAN.FindAndDoOperation(this.inv.isSelected(), model) ), "4, 2" );
		cp.add( createNumberButton("1", e -> this.model.insertDigit(1) ) ,"4, 3");
		cp.add( createNumberButton("2", e -> this.model.insertDigit(2) )  ,"4, 4");
		cp.add( createNumberButton("3", e -> this.model.insertDigit(3) )  ,"4, 5");
		cp.add( createNonNumberButton("-",e -> {
			BinaryAction action = new BinaryAction() {};
			action.defaultJob(model, (a, b) -> a-b );
		}),"4, 6");
		cp.add( createNonNumberButton("pop", e -> {
			if(stack.isEmpty()) throw new IllegalArgumentException("The stack is empty so the value can't be popped.");
			model.setValue(stack.pop());
		} )  ,"4, 7");

		// 5th row
		cp.add( createUnaryActionButton("x^n", "x^(1/n)", e -> {
			
			BinaryAction x_POW_N = new BinaryAction() { };
			x_POW_N.defaultJob(model, (a, b) -> !this.inv.isSelected() ?  pow(a, b) : pow(a, 1/b) );
			
			}), "5, 1" );
		cp.add( createUnaryActionButton("ctg", "arcctg", e -> OperationConstants.COTAN.FindAndDoOperation(this.inv.isSelected(), model) ), "5, 2" );
		cp.add( createNumberButton("0", e -> this.model.insertDigit(0)  ), "5, 3" );
		cp.add( createNonNumberButton("+/-", e -> model.swapSign() ), "5, 4" );
		cp.add( createNonNumberButton(".", e -> model.insertDecimalPoint() ), "5, 5" );
		cp.add( createNonNumberButton("+", e -> {
			BinaryAction action = new BinaryAction() {};
			action.defaultJob(model, (a, b) -> a+b );
		}), "5, 6");

		cp.add( createInvCheckBox(), "5, 7" );


	}

	/**
	 * Main method.
	 * 
	 * @param args not used
	 */
	public static void main(String[] args) {
		SwingUtilities.invokeLater(new Runnable() {
			@Override
			public void run() {
				CalculatorMain prozor = new CalculatorMain();
				prozor.setVisible(true);
			}
		});
	}

	/**
	 * Creates the number button and sets the given listener.
	 * 
	 * @param number the number on the button
	 * @param action the action listener for the button
	 * @return the created JButton object with the set parameters
	 */
	private JButton createNumberButton( String number, ActionListener action ) {
		JButton num = new JButton(number);
		num.setFont(num.getFont().deriveFont(30f));
		num.addActionListener(action);
		return num;
	}

	/**
	 * Creates the unary action button. Sets the function name, its' inverse and the given listener.
	 * 
	 * @param text		 	name of the function
	 * @param textInverse 	name of the inverse function
	 * @param action		the given action listener for the button
	 * @return the created JButton object with the set parameters
	 */
	private JButton createUnaryActionButton( String text, String textInverse, ActionListener action ) {
		JButton temp = new JButton(text);
		temp.addActionListener(action);
		ActionButton button =  new ActionButton(text, textInverse, temp);
		arrOfActionButtons.add(button);
		return temp;
	}

	/**
	 * Creates the action button. Sets the text on the button, the given listener.
	 * 
	 * @param text		 	text on the button
	 * @param action		the given action listener for the button
	 * @return the created JButton object with the set parameters
	 */
	private JButton createNonNumberButton( String text, ActionListener action ) {
		JButton button = new JButton(text);
		button.addActionListener(action);
		return button;
	}

	/**
	 * Creates a checkbox. Sets the text on the checkbox and its' listener.
	 * 
	 * @return the created JCheckBox object with the set parameters
	 */
	private JCheckBox createInvCheckBox() {
		this.inv = new JCheckBox("Inv");
		this.inv.addActionListener( e -> arrOfActionButtons.stream().forEach( button -> button.changeName() ) );
		return this.inv;
	}


	/**
	 * Class used for representing unary action 
	 * and binary action buttons with an inverse.
	 * 
	 * @author Ivan Rep
	 */
	private class ActionButton {

		private String name;
		private String invertedName;
		private JButton button;

		/**
		 * Constructor. Sets the given parameters.
		 * 
		 * @param name 			name of the function
		 * @param invertedName	name of the inverse function
		 * @param button		reference to the JButton object
		 */
		public ActionButton(String name, String invertedName, JButton button) {
			this.name = name;
			this.invertedName = invertedName;
			this.button = button;
		}

		/**
		 * Changes the name from the function name to the inverse
		 * or from the inverse to the function name. Depends on 
		 * the currently set name.
		 */
		public void changeName() {
			this.button.setText( button.getText().equals(name) ? invertedName : name );
		}

	}

}
