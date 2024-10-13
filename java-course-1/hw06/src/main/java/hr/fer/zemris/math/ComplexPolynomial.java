package hr.fer.zemris.math;

/**
 * The class represents a model for polynomials with complex roots.
 * 
 * @author Ivan Rep
 */
public class ComplexPolynomial {
	
//	public static void main(String[] args) {
//		ComplexRootedPolynomial crp = new ComplexRootedPolynomial(
//				new Complex(2,0), Complex.ONE, Complex.ONE_NEG, Complex.IM, Complex.IM_NEG
//				);
//				ComplexPolynomial cp = crp.toComplexPolynome();
//				System.out.println(crp);
//				System.out.println(cp);
//				System.out.println(cp.derive());
//
//	}
	
	private Complex[] factors;
	
	/**
	 * Constrcutor. Sets the factors arrray.
	 * 
	 * @param factors the factors array
	 */
	public ComplexPolynomial(Complex ...factors) {
		this.factors = factors;
	}

	/**
	 * Returns the order of the polynomial.
	 * 
	 * @return order of the polynomial
	 */
	public short order() {
		return (short) (factors.length-1);
	}


	/**
	 * Multiplies the current polynomial with the given polynomial and returns
	 * a new object of type ComplexPolynomial.
	 * 
	 * @param p the given polynomial
	 * @return new object of type ComplexPolynomial
	 */
	public ComplexPolynomial multiply(ComplexPolynomial p) {
		Complex[] array = new Complex[ this.order() + p.order() + 1 ];
		
		for(int i=0; i<this.factors.length; i++)
			for(int j=0; j<p.factors.length; j++)
				if(array[i+j]==null) array[i+j] = factors[i].multiply(p.factors[j]);
				else array[i+j] = array[i+j].add(factors[i].multiply(p.factors[j]));

		return new ComplexPolynomial(array);
	}

	/**
	 * Computes the first derivative of the current polynomial.
	 * 
	 * @return the first derivative
	 */
	public ComplexPolynomial derive() {
		Complex[] derivative = new Complex[factors.length-1];
		for(int i=1; i<factors.length; i++) 
			derivative[i-1] = factors[i].scale(i);
		return new ComplexPolynomial(derivative);
	}
	
	/**
	 * Computes the polynomial value at given point z.
	 * 
	 * @param z the given point z in the complex plane
	 * @return new complex number with the computed values
	 */
	public Complex apply(Complex z) {
		Complex ret = new Complex(0, 0);
		for(int i=0; i<factors.length; i++) 
			ret = ret.add( z.power(i).multiply( factors[i] ) );
		return ret;
	}
	
	/**
	 * Returns the string representation of the polynomial.
	 * 
	 * @return the string representation of the polynomial
	 */
	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		for(int i=factors.length-1; i>=0; i--) 
			sb.append( factors[i].toString()  + ( i==0 ? "" : "*z^"+i+"+" ) );
		return sb.toString();
	}

}
