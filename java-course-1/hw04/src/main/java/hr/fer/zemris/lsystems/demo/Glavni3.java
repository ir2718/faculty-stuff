package hr.fer.zemris.lsystems.demo;

import hr.fer.zemris.lsystems.gui.LSystemViewer;
import hr.fer.zemris.lsystems.impl.LSystemBuilderImpl;

/**
 * The class demonstrates the functionalities of the GUI
 * used for drawing Lindermayer systems and
 * parsing from the file.
 * 
 * @author Ivan Rep
 */
public class Glavni3 {

	/**
	 * The main method, called when the program executes.
	 * 
	 * @param args the arguments of the command line, they're not used
	 */
	public static void main(String[] args) {
		LSystemViewer.showLSystem(LSystemBuilderImpl::new);

	}

}
