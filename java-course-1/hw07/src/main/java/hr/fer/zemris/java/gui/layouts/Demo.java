package hr.fer.zemris.java.gui.layouts;

import java.awt.Container;

import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.SwingUtilities;
import javax.swing.WindowConstants;

/**
 * Demonstrates the functionality of the calculator layout.
 * 
 * @author Ivan Rep
 */
public class Demo extends JFrame {

	private static final long serialVersionUID = 1L;

	/**
	 * Creates the window and calls the initGUI method.
	 */
	public Demo() {
		super();
		setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);
		setTitle("Calc");
		setLocation(20, 20);
		setSize(500, 500);
		
		JPanel p = new JPanel();
		//p.setBorder(BorderFactory.createLineBorder(Color.BLUE, 50));
		setContentPane(p);
		
		initGUI();
	}
	
	/**
	 * Fills the GUI with a few buttons:
	 */
	private void initGUI() {
		Container cp = getContentPane();
		cp.setLayout(new CalcLayout(5));

		JLabel labela = new JLabel("Ovo je tekst!");
		JButton button = new JButton("Stisni me još jednom!");

		cp.add(new JButton("1, 1"), "1, 1");
//		cp.add(new JButton("1, 6"), "1, 6");
//		cp.add(new JButton("1, 7"), "1, 7");
//		
//		cp.add(new JButton("2, 1"), "2, 1");
//		cp.add(new JButton("2, 2"), "2, 2");
		cp.add(new JButton("2, 3"), "2, 3");
//		cp.add(new JButton("2, 4"), "2, 4");
//		cp.add(new JButton("2, 5"), "2, 5");
//		cp.add(new JButton("2, 6"), "2, 6");
		cp.add(new JButton("2, 7"), "2, 7");
//		
//		cp.add(new JButton("3, 1"), "3, 1");
//		cp.add(new JButton("3, 2"), "3, 2");
//		cp.add(new JButton("3, 3"), "3, 3");
//		cp.add(new JButton("3, 4"), "3, 4");
//		cp.add(new JButton("3, 5"), "3, 5");
//		cp.add(new JButton("3, 6"), "3, 6");
//		cp.add(new JButton("3, 7"), "3, 7");

//		cp.add(new JButton("4, 1"), "4, 1");
		cp.add(new JButton("4, 2"), "4, 2");
//		cp.add(new JButton("4, 3"), "4, 3");
//		cp.add(new JButton("4, 4"), "4, 4");
		cp.add(new JButton("4, 5"), "4, 5");
//		cp.add(new JButton("4, 6"), "4, 6");
		cp.add(new JButton("4, 7"), "4, 7");
//		
//		cp.add(new JButton("5, 1"), "5, 1");
//		cp.add(new JButton("5, 2"), "5, 2");
//		cp.add(new JButton("5, 3"), "5, 3");
//		cp.add(new JButton("5, 4"), "5, 4");
//		cp.add(new JButton("5, 5"), "5, 5");
//		cp.add(new JButton("5, 6"), "5, 6");
//		cp.add(new JButton("5, 7"), "5, 7");
		//Dimension dim = button.getPreferredSize();
		
		//labela.setBounds(10, 10, 100, 30);
		//button.setBounds(10, 50, dim.width, dim.height);
	}

	/**
	 * Main method.
	 * 
	 * @param args	not used here
	 */
	public static void main(String[] args) {
		SwingUtilities.invokeLater(new Runnable() {
			@Override
			public void run() {
				Demo prozor = new Demo();
				prozor.setVisible(true);
			}
		});
	}

}
