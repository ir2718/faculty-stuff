package hr.fer.zemris.math;

import java.util.stream.Stream;

/**
 * The class represents a model for polynomials with complex roots
 * in rooted form.
 * 
 * @author Ivan Rep
 */
public class ComplexRootedPolynomial {
	
	private Complex constant;
	private Complex[] roots;


	/**
	 * Constructor. Sets the constant and roots values to the given ones.
	 * 
	 * @param constant the given constant value
	 * @param roots the given roots values
	 */
	public ComplexRootedPolynomial(Complex constant, Complex ... roots) {
		this.constant = constant;
		this.roots = roots;
	}
	
	/** 
	 * Computes polynomial value at given point z.
	 * 
	 * @param z the given point z
	 * @return new object of type Complex with the computed values
	 */
	public Complex apply(Complex z) {
		Complex ret = constant;
		for(Complex zi : roots) 
			ret = ret.multiply( z.sub(zi) );
		return ret;
	}
	
	/**
	 * Converts this representation to ComplexPolynomial type.
	 * 
	 * @return new object of type ComplexPolynomial with the computed valuess
	 */
	public ComplexPolynomial toComplexPolynome() {
		ComplexPolynomial pol = new ComplexPolynomial(constant);
		for(Complex c : roots) 
			pol = pol.multiply( new ComplexPolynomial(c.negate(), Complex.ONE) );
		return pol;
	}
	
	/**
	 * Returns the string representation of a complex rooted polynomial.
	 * 
	 * @return the string representation of a complex rooted polynomial.
	 */
	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder(constant.toString());
		Stream.of(roots).forEach( c -> sb.append("*(z-"+c+")") );
		return sb.toString();
	}
	

	/**
	 * Finds index of closest root for given complex number z that is within
	 * threshold.
	 * if there is no such root, returns -1.
	 *
	 * @param z the given complex number
	 * @param threshold the threshold for finding the closest root
	 * @return an index of the root closest to the given complex number z
	 */ 
	 
	public int indexOfClosestRootFor(Complex z, double threshold) {
		int index = -1;
		for(int i=0; i<roots.length; i++)
			if( roots[i].sub(z).module() < threshold ) index = i;
		
		return index;
	}

}
