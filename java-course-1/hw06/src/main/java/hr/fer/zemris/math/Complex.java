package hr.fer.zemris.math;

import static java.lang.Math.*;
import java.util.ArrayList;
import java.util.List;

/**
 * The class represents a complex number and contains methods for operations on them.
 * 
 * @author Ivan Rep
 */
public class Complex {

	private double re;
	private double im;

	public static final Complex ZERO = new Complex(0,0);
	public static final Complex ONE = new Complex(1,0);
	public static final Complex ONE_NEG = new Complex(-1,0);
	public static final Complex IM = new Complex(0,1);
	public static final Complex IM_NEG = new Complex(0,-1);

	/**
	 * Empty constructor.
	 */
	public Complex() { }

	/**
	 * A constructor that takes in the real and imaginary part of a complex number.
	 * 
	 * @param re real part of a complex number
	 * @param im imaginary part of a complex number
	 */
	public Complex( double re, double im ) {
		this.re = re;
		this.im = im;
	}


	/**
	 * Method returns the module of a complex number.
	 * 
	 * @return module of the complex number
	 */
	public double module() {
		return sqrt( re*re + im*im );
	}

	/**
	 * Method returns the angle of a complex number.
	 * 
	 * @return angle of a complex number
	 */
	public double angle() {
		return (im>0 && re>0) || (im>0 && re<0) ? atan2(im, re) : (2*PI + atan2(im, re)) % (2*PI);
	}
	
	
	/**
	 * Method computes the result of multiplying two complex numbers.
	 * 
	 * @param c factor of the multiplication 
	 * @return new object of Complex class with the computed real and imaginary part
	 */
	public Complex multiply(Complex c) {
		return new Complex( this.re*c.re - this.im*c.im, this.re*c.im + this.im*c.re);
	}



	/**
	 * Method computes the result of dividing two complex numbers.
	 * 
	 * @param c denominator
	 * @return new object of ComplexNumber class with the computed real and imaginary part
	 */
	public Complex divide(Complex c) {
		return new Complex( ( re*c.re+ im*c.im )/( c.re*c.re+c.im*c.im), 
				( im*c.re - re*c.im )/( c.re*c.re+c.im*c.im ) );
	}


	/**
	 * Method computes the result of adding two complex numbers.
	 * 
	 * @param c number that is added
	 * @return new object of class Complex with the computed real and imaginary part
	 */
	public Complex add(Complex c) {
		return new Complex( this.re + c.re, this.im + c.im );
	}



	/**
	 * Method computes the result of subtracting two complex numbers.
	 * 
	 * @param c diminutive of the subtraction
	 * @return new object of class Complex with the computed real and imaginary part
	 */
	public Complex sub(Complex c) {
		return new Complex( this.re - c.re, this.im - c.im );
	}



	/**
	 * Method returns a new complex number with negative real and imaginary part.
	 * 
	 * @return new object of class Complex with the negative real and imaginary part
	 */
	public Complex negate() {
		return new Complex( -this.re, -this.im );
	}


	/**
	 * Method computes the result of exponentiation of a complex number.
	 * 
	 * @param n the exponent
	 * @return new object of Complex class with the computed real and imaginary part
	 * @throws IllegalArgumentException if the given n is less than 0
	 */
	public Complex power(int n) {
		if(n<0) throw new IllegalArgumentException("The power must not be less than 0. The given power was "+n+".");
		return new Complex( pow(module(), n)*cos(n*angle()),
				pow(module(), n)*sin(n*angle()));
	}


	/**
	 * Method computes the nth root of a complex number.
	 * 
	 * @param n roots of this exponent are calculated
	 * @return new object of Complex class with the computed real and imaginary part
	 * @throws IllegalArgumentException if the given n is less or equal to 0
	 */
	public List<Complex> root(int n) {
		if(n<=0) throw new IllegalArgumentException("The root must not be less than or equal to 0. The given root was "+n+".");
		List<Complex> list = new ArrayList<>();
		
		for(int i=0; i<n; i++) 
			list.add( new Complex( pow( module(), 1.0/n)*cos( ( angle()+2*i*PI )/n ) ,
					pow(module(), 1.0/n)*sin( ( angle()+2*i*PI )/n ) ) );

		return list;
		
	}
	
	/**
	 * Method scales the real and imaginary part of the current complex number
	 * by the given scalar.
	 * 
	 * @param scalar the given scalar
	 * @return a new object of Complex class with the scaled real and imaginary part
	 */
	public Complex scale(int scalar) {
		return new Complex( this.re*scalar, this.im*scalar );
	}


	/**
	 * Returns the string representation of a complex number.
	 * 
	 * @return teh string representation of a complex number
	 */
	@Override
	public String toString() {
		return "(" + ( im>=0 ?  re+"+i"+im : re+"-i"+abs(im) ) + ")" ;
	}

}
