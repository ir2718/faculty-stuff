package hr.fer.zemris.java.gui.charts;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Component;
import java.awt.Container;
import java.awt.Graphics;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;

import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.SwingConstants;
import javax.swing.SwingUtilities;
import javax.swing.WindowConstants;

/**
 * Demonstrates the bar chart functionality.
 * 
 * @author Ivan Rep
 */
public class BarChartDemo extends JFrame {

	private static final long serialVersionUID = 1L;
	private static BarChart model;
	private static String label;

	/**
	 * Constructor. Executes the InitGui method
	 * and creates the window.
	 */
	public BarChartDemo() {
		setLocation(20, 50);
		setSize(550, 550);
		setTitle("Bar Chart");
		setBackground(Color.white);
		setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);
		//		model = new BarChart(
		//				 Arrays.asList(
		//						 new XYValue(1,8), new XYValue(2,20), new XYValue(3,22),
		//						 new XYValue(4,10), new XYValue(5,4)
		//						 ),
		//						 "Number of people in the car",
		//						 "Frequency",
		//						 0, // y-os kreće od 0
		//						 22, // y-os ide do 22
		//						 2
		//						);
		initGUI();
	}

	/**
	 * Initializes the GUI.
	 */
	private void initGUI() {
		Container cp = getContentPane();
		cp.setLayout(new BorderLayout());
		cp.add(new JLabel(label, SwingConstants.CENTER), BorderLayout.NORTH);
		cp.add(new BarChartComponent(model), BorderLayout.CENTER);
	}


	/**
	 * Main method. Uses a file from the command line args to
	 * draw the bar chart.
	 * 
	 * @param args						path to file with the information for the bar chart
	 * @throws IOException				if the given file doesn't exist
	 * @throws IllegalArgumentException	if the args length isn't equal to one
	 */
	public static void main(String[] args) throws IOException {

		if(args.length!=1) throw new IllegalArgumentException("The program expects exactly one argument.");

		File f = new File(args[0]);
		model = BarChartFactory.parse(f);
		label = args[0];

		SwingUtilities.invokeLater(() -> {
			JFrame frame = new BarChartDemo();
			frame.setVisible(true);
		});
	}



}
