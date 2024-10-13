package hr.fer.zemris.java.fractals;

import java.util.ArrayList;
import java.util.Scanner;
import java.util.concurrent.atomic.AtomicBoolean;
import hr.fer.zemris.java.fractals.viewer.FractalViewer;
import hr.fer.zemris.java.fractals.viewer.IFractalProducer;
import hr.fer.zemris.java.fractals.viewer.IFractalResultObserver;
import hr.fer.zemris.math.Complex;
import hr.fer.zemris.math.ComplexPolynomial;
import hr.fer.zemris.math.ComplexRootedPolynomial;

/**
 * Demonstrates the functionalities of the Complex, ComplexPolynomial and
 * ComplexRootedPolynomial classes. Uses GUI for drawing.
 * 
 * @author Ivan Rep
 */
public class Newton {

	private static ArrayList<Complex> lines = new ArrayList<>();
	private static final double ROOT_THRESHOLD = 0.002;
	private static final double CONVERGENCE_THRESHOLD = 0.001;
	private static final int CONSTANT_16= 16;

	/**
	 * Main method.
	 * 
	 * @param args not used here
	 */
	public static void main(String[] args) {


		System.out.println("Welcome to Newton-Raphson iteration-based fractal viewer.\n"
				+ "Please enter at least two roots, one root per line. Enter 'done' when done.");

		Scanner sc = new Scanner(System.in);
		readInput(sc, lines);
		sc.close();
		
		FractalViewer.show(new SomeProducer());

	}

	/**
	 * Method reads the input from the command line and writes
	 * the current row number.
	 * 
	 * @param sc Scanner object used for reading input
	 * @param lines list used for storing complex numbers
	 */
	protected static void readInput(Scanner sc, ArrayList<Complex> lines) {

		int row = 1;
		while(true) {
			System.out.print("Root "+row+"> ");
			String line = null;
			while(line==null) {
				line = sc.nextLine();
				if(line.equals("done")) break;
				try {
					parse(line, lines);
				} catch(Exception exc) {
					throw new IllegalArgumentException("The given input is incorrectly formatted.");
				}
				row++;
			}
			if(line.equals("done")) break;
		}
	}

	/**
	 * Parses the input from the console.
	 * 
	 * @param line a line from the input
	 * @return new object of type Complex 
	 */
	protected static void parse(String line, ArrayList<Complex> lines) {

		line = line.trim();
		String help = line.substring(1, line.length());
		help = help.replace("+", " + ").replace("-", " - ");
		String[] s = ( line.substring(0, 1) + help ).split("\\s+");
		
		Complex c = null;

		if(s.length==1 ) {
			if( s[0].contains("i") )
				if(s[0].equals("i") || s[0].equals("+i") ) c = new Complex(0,1);
				else if(s[0].equals("-i")) c = new Complex(0, -1);
				else c = new Complex( 0, Double.parseDouble(s[0].replace("i", "") ) );
			else
				c = new Complex( Double.parseDouble(s[0]), 0 );
		} else if(s.length==2) {
			if(s[0].contains("+") && s[1].contains("i")) c = new Complex(0, 1);
			else if(s[0].contains("-") && s[1].contains("i")) c = new Complex(0, -1);
			else if(s[0].contains("+") ) c = new Complex( Double.parseDouble(s[1]), 0 );
			else if(s[0].contains("-") ) c = new Complex( -Double.parseDouble(s[1]), 0 );
		} else if(s.length==3) {
			if(s[2].equals("i") && s[1].equals("+"))  c = new Complex( Double.parseDouble(s[0]),  1 );
			else if(s[2].equals("i") && s[1].equals("-"))  c = new Complex( Double.parseDouble(s[0]),  -1 );
			else c = new Complex( Double.parseDouble(s[0]),  Double.parseDouble( (s[1].equals("+") ? "" : "-" ) + s[2].replace("i", ""))  );
		} else if(s.length==4) {
			if(s[0].equals("+")) 
				switch(s[2]) {
				case "+" -> c = new Complex( Double.parseDouble(s[1]), Double.parseDouble( s[3].replace("i", "")) );
				case "-" -> c = new Complex( Double.parseDouble(s[1]), -Double.parseDouble( s[3].replace("i", "")) );
				}
			else if(s[0].equals("-"))
				switch(s[2]) {
				case "+" -> c = new Complex( -Double.parseDouble(s[1]), Double.parseDouble( s[3].replace("i", "")) );
				case "-" -> c = new Complex( -Double.parseDouble(s[1]), -Double.parseDouble( s[3].replace("i", "")) );
				}
		}	
			
		lines.add(c);

	}

	/**
	 * Class is used for drawing on the GUI and the main logic behind the drawing. 
	 * 
	 * @author Ivan Rep
	 */
	public static class SomeProducer implements IFractalProducer {

		/**
		 * Method for drawing on the GUI and the main logic behind the drawing.
		 */
		@Override
		public void produce(double reMin, double reMax, double imMin, double imMax,
				int width, int height, long requestNo, IFractalResultObserver observer, AtomicBoolean cancel) {
			System.out.println("Zapocinjem izracun...");
			int m = CONSTANT_16*CONSTANT_16*CONSTANT_16;
			int index = 0;
			int offset = 0;
			short[] data = new short[width * height];
			Complex[] arr = new Complex[lines.size()];
			for(int i=0; i<arr.length; i++) arr[i] = lines.get(i);
			ComplexRootedPolynomial rcp = new ComplexRootedPolynomial( Complex.ONE, arr );
			ComplexPolynomial cp = rcp.toComplexPolynome();
			for(int y = 0; y < height; y++) {
				if(cancel.get()) break;
				for(int x = 0; x < width; x++) {
					double cre = x / (width-1.0) * (reMax - reMin) + reMin;
					double cim = (height-1.0-y) / (height-1) * (imMax - imMin) + imMin;
					double module = 0;
					Complex c = new Complex(cre, cim);
					Complex zn = c;
					int iters = 0;
					do {
						Complex numerator = cp.apply(zn);
						Complex denominator = (cp.derive()).apply(zn);
						Complex znold = zn;
						Complex fraction = numerator.divide(denominator);
						zn = zn.sub(fraction);
						module = znold.sub(zn).module();
						iters++;
					} while( module > CONVERGENCE_THRESHOLD && iters < m);
					index = rcp.indexOfClosestRootFor(zn, ROOT_THRESHOLD);
					data[offset]=(short) (index+1);
					offset++;
				}

			}
			System.out.println("Racunanje gotovo. Idem obavijestiti promatraca tj. GUI!");
			observer.acceptResult(data, (short)(cp.order()+1), requestNo);
		}
	}


}

