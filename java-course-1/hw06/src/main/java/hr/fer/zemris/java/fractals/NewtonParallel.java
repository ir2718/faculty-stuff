package hr.fer.zemris.java.fractals;

import java.util.ArrayList;
import java.util.Scanner;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.atomic.AtomicBoolean;
import hr.fer.zemris.java.fractals.viewer.*;
import hr.fer.zemris.math.Complex;
import hr.fer.zemris.math.ComplexPolynomial;
import hr.fer.zemris.math.ComplexRootedPolynomial;

/**
 * Demonstrates the functionalities of the Complex, ComplexPolynomial and
 * ComplexRootedPolynomial classes. Uses GUI for drawing and multithreading 
 * for computing the values.
 * 
 * @author Ivan Rep
 */
public class NewtonParallel {

	private static ArrayList<Complex> lines = new ArrayList<>();
	private static final double ROOT_THRESHOLD = 0.002;
	private static final double CONVERGENCE_THRESHOLD = 0.001;
	private static final int CONSTANT_16= 16;
	private static int workers = Runtime.getRuntime().availableProcessors();
	private static int tracks = 4*Runtime.getRuntime().availableProcessors();
	
	private static Complex[] arr;
	private static ComplexRootedPolynomial rcp;
	private static ComplexPolynomial cp;

	/**
	 * Main method. Uses command line arguments to determine number 
	 * of threads and tracks.
	 * 
	 * @param args workers and tracks arguments
	 */
	public static void main(String[] args) {

		for(int i=0; i<args.length; i++) {

			String s = args[i].trim();

			try {
				
				if( s.startsWith("--workers=") ) {
					s = s.replace("--workers=", "");
					if(Integer.parseInt(s)<1) throw new IllegalArgumentException(); 
					workers = Integer.parseInt(s);
				} else if( s.startsWith("--tracks=") ) {
					s = s.replace("--tracks=", "");
					if(Integer.parseInt(s)<1) throw new IllegalArgumentException(); 
					tracks = Integer.parseInt(s);
				} else if( s.equals("-w") || s.equals("-t") ) {
					i++;
					String num = args[i].trim();
					if(s.equals("-w"))
						if(Integer.parseInt(num)<1) throw new IllegalArgumentException(); 
						else workers = Integer.parseInt(num);
					else if(s.equals("-t")) 
						if(Integer.parseInt(num)<1) throw new IllegalArgumentException();
						else tracks = Integer.parseInt(num);
				} else { throw new IllegalArgumentException(); }
				
			} catch(Exception exc) {
				throw new IllegalArgumentException("The input is incorrectly formatted or forbidden.");
			}

		}


		System.out.println("Welcome to Newton-Raphson iteration-based multithreaded fractal viewer.\n"
				+ "Please enter at least two roots, one root per line. Enter 'done' when done.");

		Scanner sc = new Scanner(System.in);
		Newton.readInput(sc, lines);
		sc.close();
		
		
		arr = new Complex[lines.size()];
		for(int i=0; i<arr.length; i++) arr[i] = lines.get(i);
		rcp = new ComplexRootedPolynomial( Complex.ONE, arr );
		cp = rcp.toComplexPolynome();

		System.out.println(lines);
		FractalViewer.show(new ThreadProducer());	

	}


	/**
	 * Class used for computing the values used in drawing on the GUI.
	 * 
	 * @author Ivan Rep
	 */
	public static class PosaoIzracuna implements Runnable {
		double reMin;
		double reMax;
		double imMin;
		double imMax;
		int width;
		int height;
		int yMin;
		int yMax;
		int m;
		short[] data;
		AtomicBoolean cancel;
		public static PosaoIzracuna NO_JOB = new PosaoIzracuna();
		
		/**
		 * Empty constructor.
		 */
		private PosaoIzracuna() {
		}
		
		/**
		 * Constructor sets the values to the given values.
		 * 
		 * @param reMin the given reMin
		 * @param reMax the given reMax
		 * @param imMin the given imMin
		 * @param imMax the given imMax
		 * @param width the given width
		 * @param height the given height
		 * @param yMin the given yMin
		 * @param yMax the given yMax
		 * @param m the given m
		 * @param data the given data
		 * @param cancel the given cancel
		 */
		public PosaoIzracuna(double reMin, double reMax, double imMin,
				double imMax, int width, int height, int yMin, int yMax, 
				int m, short[] data, AtomicBoolean cancel) {
			super();
			this.reMin = reMin;
			this.reMax = reMax;
			this.imMin = imMin;
			this.imMax = imMax;
			this.width = width;
			this.height = height;
			this.yMin = yMin;
			this.yMax = yMax;
			this.m = m;
			this.data = data;
			this.cancel = cancel;
		}

		@Override
		public void run() {
			
			int m = CONSTANT_16*CONSTANT_16*CONSTANT_16;
			int index = 0;
			for(int y = yMin; y <= yMax; y++) {
				if(cancel.get()) break;
				for(int x = 0; x < width; x++) {
					double cre = x / (width-1.0) * (reMax - reMin) + reMin;
					double cim = (height-1.0-y) / (height) * (imMax - imMin) + imMin;
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
					data[y*width+x]=(short) (index+1);
				}
			}
		
		}
	}

	/**
	 * Class used for creating the threads and running them.
	 * 
	 * @author Ivan Rep
	 */
	public static class ThreadProducer implements IFractalProducer {
		
		
		/**
		 * Method creates the threads and runs them.
		 */
		@Override
		public void produce(double reMin, double reMax, double imMin, double imMax,
				int width, int height, long requestNo, IFractalResultObserver observer, AtomicBoolean cancel) {
			System.out.println("Zapocinjem izracun...");
			int m = CONSTANT_16*CONSTANT_16*CONSTANT_16;
			short[] data = new short[width * height];
			
			if(tracks>height) tracks = height;
			final int brojTraka = tracks;
			
			int brojYPoTraci = height / brojTraka;

			final BlockingQueue<PosaoIzracuna> queue = new LinkedBlockingQueue<>();

			System.out.println("Number of threads: "+workers+"\nNumber of jobs: "+brojTraka);

			Thread[] radnici = new Thread[workers];
			for(int i = 0; i < radnici.length; i++) {
				radnici[i] = new Thread(new Runnable() {
					@Override
					public void run() {
						while(true) {
							PosaoIzracuna p = null;
							try {
								p = queue.take();
								if(p==PosaoIzracuna.NO_JOB) break;
							} catch (InterruptedException e) {
								continue;
							}
							p.run();
						}
					}
				});
			}
			for(int i = 0; i < radnici.length; i++) {
				radnici[i].start();
			}
			
			for(int i = 0; i < brojTraka; i++) {
				int yMin = i*brojYPoTraci;
				int yMax = (i+1)*brojYPoTraci-1;
				if(i==brojTraka-1) {
					yMax = height-1;
				}
				PosaoIzracuna posao = new PosaoIzracuna(reMin, reMax, imMin, imMax, width, height, yMin, yMax, m, data, cancel);
				while(true) {
					try {
						queue.put(posao);
						break;
					} catch (InterruptedException e) {
					}
				}
			}
			for(int i = 0; i < radnici.length; i++) {
				while(true) {
					try {
						queue.put(PosaoIzracuna.NO_JOB);
						break;
					} catch (InterruptedException e) {
					}
				}
			}
			
			for(int i = 0; i < radnici.length; i++) {
				while(true) {
					try {
						radnici[i].join();
						break;
					} catch (InterruptedException e) {
					}
				}
			}
			
			System.out.println("Racunanje gotovo. Idem obavijestiti promatraca tj. GUI!");
			observer.acceptResult(data, (short)(cp.order()+1), requestNo);
		}
	}
	

}
