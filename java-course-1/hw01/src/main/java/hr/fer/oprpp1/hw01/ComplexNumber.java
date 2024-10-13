package hr.fer.oprpp1.hw01;

import static java.lang.Math.*;

/**
 * The class represents a complex number and contains methods for operations on them.
 * 
 * @author Ivan Rep
 */
public class ComplexNumber {

	private double re;
	private double im;

	/**
	 * A constructor that takes in the real and imaginary part of a complex number.
	 * 
	 * @param re the real part of a complex number
	 * @param im the imaginary part of a complex number
	 */
	public ComplexNumber(double re, double im) {
		this.re = re;
		this.im = im;
	}

	/**
	 * The method returns a new complex number with the given real part and an imaginary part of 0.
	 * 
	 * @param real the real part assigned to the complex number
	 * @return a new object of the ComplexNumber class with the given real part and an imaginary part set to 0
	 */
	public static ComplexNumber fromReal(double real) {
		return new ComplexNumber(real, 0);
	}

	/**
	 * The method returns a new complex number with the real part set to 0 and the imaginary part set to the given value.
	 * 
	 * @param imaginary the imaginary part assigned to the complex number
	 * @return a new object of the ComplexNumber class with the given imaginary part and a real part set to 0
	 */
	public static ComplexNumber fromImaginary(double imaginary) {
		return new ComplexNumber(0, imaginary);
	}

	/**
	 * The method transforms a complex number from its polar coordinates to the cartesian coordinates.
	 * 
	 * @param magnitude the radius or magnitude of the complex number in trigonometric form
	 * @param angle the angle of the complex number in trigonometric form
	 * @return a new object of the ComplexNumber class with the computed cartesian coordinates 
	 */
	public static ComplexNumber fromMagnitudeAndAngle(double magnitude, double angle) {
		return new ComplexNumber( magnitude*cos(angle), magnitude*sin(angle) );
	}

	/**
	 * The method parses the input and returns a complex number that corresponds to the given input. 
	 * 
	 * @param s the String that is parsed
	 * @return a new ComplexNumber object that corresponds to the given input
	 * @throws NumberFormatException if the String can't be parsed
	 */
	public static ComplexNumber parse(String s) {

		int lastIndex = s.lastIndexOf('i');
		
		if(lastIndex == -1) 
			try { return new ComplexNumber( Double.parseDouble(s), 0 ); } // if the number only consists of only the real part
			catch (NumberFormatException exc) {}

		else{
			char[] arr = s.toCharArray();

			int i = arr[0]=='+' || arr[0]=='-' ? 1 : 0  ;
			while( Character.isDigit(arr[i]) || arr[i]=='.') {
				if(arr[i+1]=='E' && (arr[i+2]=='-'||arr[i+2]=='+') ) i+=2;
				else if(arr[i+1]=='E' && Character.isDigit(arr[i+2]) ) i+=2;
				i++;
			}

			if(arr[i]=='i') { // if the number consists of only the imaginary part

				// if the given value is "i"
				if(s.length()==1) 
					return new ComplexNumber(0, 1);

				//if the given value is "+i" or "+i"
				else if( s.length()==2 && (arr[0]=='-' || arr[0]=='+') )
					return arr[0]=='-' ? new ComplexNumber(0, -1) : new ComplexNumber(0, 1) ;

				try { return new ComplexNumber( 0, Double.parseDouble(s.substring(0, i)) ); }
				catch (NumberFormatException exc) {throw new NumberFormatException("The input can't be parsed."); }
			} 
			else if (arr[i]=='-' || arr[i]=='+') { // if the number has both parts

				if( s.contains("+i") )
					return new ComplexNumber( Double.parseDouble( s.substring(0, i) ), 1 );

				else if( s.contains("-i") )
					return new ComplexNumber( Double.parseDouble( s.substring(0, i) ), -1);

				try { return new ComplexNumber( Double.parseDouble( s.substring(0, i) ), Double.parseDouble( s.substring(i, s.length()-1) ) ); }
				catch (NumberFormatException exc) {throw new NumberFormatException("The input can't be parsed."); }
			}
		}

		throw new NumberFormatException("The input can't be parsed");
	}


	/**
	 * The getter method for the real part of a complex number.
	 * 
	 * @return the real part of the complex number
	 */
	public double getReal() {
		return this.re;
	}

	/**
	 * The getter method for the imaginary part of a complex number.
	 * 
	 * @return the imaginary part of the complex number
	 */
	public double getImaginary() {
		return this.im;
	}

	/**
	 * The getter method for the magnitude of a complex number.
	 * 
	 * @return the radius or magnitude of the complex number
	 */
	public double getMagnitude() {
		return sqrt( re*re + im*im );
	}

	/**
	 * The getter method for the angle of a complex number.
	 * 
	 * @return the angle of a complex number
	 */
	public double getAngle() {
		return (im>0 && re>0) || (im>0 && re<0) ? atan2(im, re) : (2*PI + atan2(im, re)) % (2*PI);
	}

	/**
	 * The method computes the result of adding two complex numbers.
	 * 
	 * @param c the number that is added
	 * @return a new object of ComplexNumber class with the computed real and imaginary part
	 */
	public ComplexNumber add(ComplexNumber c) {
		return new ComplexNumber( this.re + c.re, this.im + c.im );
	}

	/**
	 * The method computes the result of subtracting two complex numbers.
	 * 
	 * @param c the diminutive of the subtraction
	 * @return a new object of ComplexNumber class with the computed real and imaginary part
	 */
	public ComplexNumber sub(ComplexNumber c) {
		return new ComplexNumber( this.re - c.re, this.im - c.im );
	}

	/**
	 * The method computes the result of multiplying two complex numbers.
	 * 
	 * @param c the factor of the multiplication 
	 * @return a new object of ComplexNumber class with the computed real and imaginary part
	 */
	public ComplexNumber mul(ComplexNumber c) {
		return new ComplexNumber( this.re*c.re - this.im*c.im, this.re*c.im + this.im*c.re);
	}

	/**
	 * The method computes the result of dividing two complex numbers.
	 * 
	 * @param c the denominator
	 * @return a new object of ComplexNumber class with the computed real and imaginary part
	 */
	public ComplexNumber div(ComplexNumber c) {
		return new ComplexNumber( ( re*c.re+ im*c.im )/( c.re*c.re+c.im*c.im),
				( im*c.re - re*c.im )/( c.re*c.re+c.im*c.im ) );
	}

	/**
	 * The method computes the result of exponentiation of a complex number.
	 * 
	 * @param n the exponent
	 * @return a new object of ComplexNumber class with the computed real and imaginary part
	 * @throws IllegalArgumentException if the given n is less than 0
	 */
	public ComplexNumber power(int n) {
		if(n<0) throw new IllegalArgumentException("The power must not be less than 0! The given power was "+n+".");
		return new ComplexNumber( pow(getMagnitude(), n)*cos(n*getAngle()), pow(getMagnitude(), n)*sin(n*getAngle()));
	}

	/**
	 * The method computes the nth root of a complex number.
	 * 
	 * @param n roots of this exponent are calculated
	 * @return a new object of ComplexNumber class with the computed real and imaginary part
	 * @throws IllegalArgumentException if the given n is less or equal to 0
	 */
	public ComplexNumber[] root(int n) {
		if(n<=0) throw new IllegalArgumentException("The root must not be less or equal to 0! The given root was "+n+".");
		ComplexNumber[] array = new ComplexNumber[n];
		
		for(int i=0; i<n; i++) 
			array[i] = new ComplexNumber( pow(getMagnitude(), 1.0/n)*cos( ( getAngle()+2*i*PI )/n ) ,
					pow(getMagnitude(), 1.0/n)*sin( ( getAngle()+2*i*PI )/n ) );

		return array;
	}

	/**
	 * The method computes and returns a String representation of a complex number. 
	 * 
	 * @return String representation of a ComplexNumber object in the format "a+bi"
	 */
	public String toString() {
		return im>=0 ? re+"+"+im+"i" : re+""+im+"i" ;
	}
}
