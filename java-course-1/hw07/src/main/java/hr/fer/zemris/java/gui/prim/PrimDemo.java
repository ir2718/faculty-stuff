package hr.fer.zemris.java.gui.prim;

import java.awt.BorderLayout;
import java.awt.Container;
import java.awt.Dimension;
import java.awt.GridLayout;
import java.awt.event.AdjustmentEvent;
import java.awt.event.AdjustmentListener;

import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JList;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.SwingUtilities;
import javax.swing.WindowConstants;

/**
 * Class demonstrates functionalities of the PrimListModel class.
 * 
 * @author Ivan Rep
 */
public class PrimDemo extends JFrame {

	private static final long serialVersionUID = 1L;
	
	private PrimListModel<Integer> model;

	/**
	 * Constructor. Creates the window for GUI.
	 */
	public PrimDemo() {
		setLocation(50, 50);
		setSize(500, 500);
		setTitle("PrimDemo");
		setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);
		initGUI();
	}
	
	/**
	 * Fills the GUI with two lists and a button.
	 */
	private void initGUI() {
		Container cp = getContentPane();
		cp.setLayout(new BorderLayout());
		
		model = new PrimListModel<>();
		model.add(model.getCurrPrime());		
		

		JPanel centrePanel = new JPanel(new GridLayout(1, 0));
		
		JList<Integer> list1 = new JList<>(model);
		JScrollPane p1 = new JScrollPane(list1, JScrollPane.VERTICAL_SCROLLBAR_AS_NEEDED, JScrollPane.HORIZONTAL_SCROLLBAR_AS_NEEDED);
		p1.setPreferredSize(new Dimension(300, 300));
		centrePanel.add(p1);
		
		JList<Integer> list2 = new JList<>(model);
		JScrollPane p2 = new JScrollPane(list2, JScrollPane.VERTICAL_SCROLLBAR_AS_NEEDED, JScrollPane.HORIZONTAL_SCROLLBAR_AS_NEEDED);
		p2.setPreferredSize(new Dimension(300, 300));
		centrePanel.add(p2);

		JButton next = new JButton("Sljedeći");
		next.addActionListener(e -> model.add(model.next()) );
		cp.add(centrePanel, BorderLayout.CENTER);
		cp.add(next, BorderLayout.PAGE_END);

	}

	/**
	 * Main method.
	 * 
	 * @param args not used here
	 */
	public static void main(String[] args) {

		SwingUtilities.invokeLater(() -> {
			JFrame frame = new PrimDemo();
			frame.pack();
			frame.setVisible(true);
		});
	}	
	
}
