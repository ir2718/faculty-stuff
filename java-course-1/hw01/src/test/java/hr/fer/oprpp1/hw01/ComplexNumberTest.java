package hr.fer.oprpp1.hw01;

import static org.junit.jupiter.api.Assertions.*;

import java.text.DecimalFormat;

import static hr.fer.oprpp1.hw01.ComplexNumber.*;
import static java.lang.Math.*;
import org.junit.jupiter.api.Test;

public class ComplexNumberTest {
	
	@Test
	public void testFirstConstructor() {
		ComplexNumber c = new ComplexNumber(1, -3);
		assertEquals( 1, c.getReal(), "The real part of the complex number should be 1! It is "+c.getReal()+".");
		assertEquals( -3, c.getImaginary(), "The imaginary part of the complex number should be -3! It is "+c.getImaginary()+".");
	} 
	
	@Test
	public void testFromRealMethod() {
		ComplexNumber c = ComplexNumber.fromReal(2.32);
		assertEquals( 2.32, c.getReal(), "The real part of the complex number should be 2.32! It is "+c.getReal()+".");
		assertEquals( 0, c.getImaginary(), "The imaginary part of the complex number should be 0! It is "+c.getImaginary()+".");
	} 
	
	@Test
	public void testFromImaginaryMethod() {
		ComplexNumber c = ComplexNumber.fromImaginary(2.32);
		assertEquals( 0, c.getReal(), "The real part of the complex number should be 0! It is "+c.getReal()+".");
		assertEquals( 2.32, c.getImaginary(), "The imaginary part of the complex number should be 2.32! It is "+c.getImaginary()+".");
	}
	
	@Test
	public void testFromMagnitudeAndAngleMethod() {
		ComplexNumber c = ComplexNumber.fromMagnitudeAndAngle(5.44, 5.34);
		DecimalFormat df = new DecimalFormat("#.##");
		assertEquals( "3.19", df.format(c.getReal()), "The real part should be 3.19 but it's "+df.format(c.getReal())+"." );
		assertEquals( "-4.4", df.format(c.getImaginary()), "The imaginary part should be -4.40 but it's "+df.format(c.getImaginary())+"." );
	}
	
	@Test
	public void testGetRealMethod() {
		ComplexNumber c = new ComplexNumber(1.4,5.3);
		assertEquals( 1.4, c.getReal(), "The real part of the complex number should be 1.4 but it is "+c.getReal()+"." );
	}
	
	@Test
	public void testGetImaginaryMethod() {
		ComplexNumber c = new ComplexNumber(1.4,5.3);
		assertEquals( 5.3, c.getImaginary(), "The imaginary part of the complex number should be 5.3 but it is "+c.getImaginary()+"." );
	}

	@Test
	public void testGetAngleMethod() {
		ComplexNumber c1 = new ComplexNumber(1,1);
		ComplexNumber c2 = new ComplexNumber(1, -1);
		ComplexNumber c3 = new ComplexNumber(-1, -1);
		assertEquals( true, abs(c1.getAngle()-0.78539)<0.001 , "The angle should be 0.78539 but it is "+c1.getAngle()+"." );
		assertEquals( true, abs(c2.getAngle()-5.4977)<0.001 , "The angle should be 5.4977 but it is "+c2.getAngle()+"." );
		assertEquals( true, abs(c3.getAngle()-3.92699)<0.001 , "The angle should be 3.92699 but it is "+c3.getAngle()+"." );
	}
	
	@Test
	public void testAddMethod() {
		ComplexNumber c1 = new ComplexNumber(1,1);
		ComplexNumber c2 = new ComplexNumber(1, -1);
		ComplexNumber c3 = c1.add(c2);
		assertEquals( 2 ,c3.getReal(),"The real part should be 2 but it is "+c3.getReal()+".");
		assertEquals( 0 ,c3.getImaginary(),"The imaginary part should be 0 but it is "+c3.getImaginary()+".");
	}
	
	@Test
	public void testSubMethod() {
		ComplexNumber c1 = new ComplexNumber(1,1);
		ComplexNumber c2 = new ComplexNumber(1, -1);
		ComplexNumber c3 = c1.sub(c2);
		assertEquals( 0 ,c3.getReal(),"The real part should be 0 but it is "+c3.getReal()+".");
		assertEquals( 2 ,c3.getImaginary(),"The imaginary part should be -2 but it is "+c3.getImaginary()+".");
	}
	
	@Test
	public void testMulMethod() {
		ComplexNumber c1 = new ComplexNumber(1,5);
		ComplexNumber c2 = new ComplexNumber(-2, 3);
		ComplexNumber c3 = c1.mul(c2);
		assertEquals( -17, c3.getReal(),"The real part should be -17 but it is "+c3.getReal()+".");
		assertEquals( -7, c3.getImaginary(),"The imaginary part should be -7 but it is "+c3.getImaginary()+".");
	}
	
	@Test
	public void testDivMethod() {
		ComplexNumber c1 = new ComplexNumber(1,5);
		ComplexNumber c2 = new ComplexNumber(-2, 3);
		ComplexNumber c3 = c1.div(c2);
		assertEquals( 1, c3.getReal(),"The real part should be 1 but it is "+c3.getReal()+".");
		assertEquals( -1, c3.getImaginary(),"The imaginary part should be -1 but it is "+c3.getImaginary()+".");
	}
	
	@Test
	public void testPowerMethod() {
		ComplexNumber c1 = new ComplexNumber(1,-3);
		ComplexNumber c3 = c1.power(3);
		assertEquals( true, abs(-26 - c3.getReal())<0.01 ,"The real part should be -26 but it is "+c3.getReal()+".");
		assertEquals( true, abs(18 - c3.getImaginary())<0.01 ,"The imaginary part should be 18 but it is "+c3.getImaginary()+".");
	}
	
	@Test
	public void testNthRoot() {
		ComplexNumber num1 = new ComplexNumber(2, 9).root(3)[2];
		ComplexNumber num2 = new ComplexNumber(-4.2, 2.6).root(5)[4];
		assertEquals( true, abs( -0.152704-num1.getReal() )<0.001, "The expected value of the real part was -0.152704 but the computed value is "+num1.getReal()+"." );
		assertEquals( true, abs( -0.927243-num2.getImaginary() )<0.001, "The expected value of the imaginary part was -0.92724 but the computed value is "+num2.getImaginary()+"." );
	}
	
	@Test
	public void testToStringMethod() {
		ComplexNumber num1 = new ComplexNumber(2.12, 9.412);
		ComplexNumber num2 = new ComplexNumber(-2.12, 9.412);
		assertEquals( "2.12+9.412i", num1.toString() , "The expected output is 2.12+9.412i but the computed value was "+num1.toString()+"." );
		assertEquals( "-2.12+9.412i", num2.toString(), "The expected output is 2.12+9.412i but the computed value was "+num1.toString()+"." );
	}
	
	@Test
	public void testParseMethodWithOnlyOneArg() {
		ComplexNumber num1 = parse("321");
		ComplexNumber num2 = parse("-321");
		ComplexNumber num3 = parse("3.21");
		ComplexNumber num4 = parse("-3.21");
		ComplexNumber num5 = parse("321i");
		ComplexNumber num6 = parse("-321i");
		ComplexNumber num7 = parse("3.21i");
		ComplexNumber num8 = parse("-3.21i");
		ComplexNumber num9 = parse("i");
		ComplexNumber num10 = parse("+i");
		ComplexNumber num11 = parse("-i");
		assertEquals( 321.0, num1.getReal(), "The real part should be 321 but it is "+num1.getReal()+"." );
		assertEquals( -321.0, num2.getReal(), "The real part should be -321 but it is "+num2.getReal()+"." );
		assertEquals( 3.21, num3.getReal(), "The real part should be 3.21 but it is "+num3.getReal()+"." );
		assertEquals( -3.21, num4.getReal(), "The real part should be -3.21 but it is "+num4.getReal()+"." );
		assertEquals( 321.0, num5.getImaginary(), "The imaginary part should be 321 but it is "+num5.getImaginary()+"." );
		assertEquals( -321.0, num6.getImaginary(), "The imaginary part should be -321 but it is "+num6.getImaginary()+"." );
		assertEquals( 3.21, num7.getImaginary(), "The imaginary part should be 3.21 but it is "+num7.getImaginary()+"." );
		assertEquals( -3.21, num8.getImaginary(), "The imaginary part should be -3.21 but it is "+num8.getImaginary()+"." );
		assertEquals( 1, num9.getImaginary(), "The imaginary part should be 1 but it is "+num9.getImaginary()+"." );
		assertEquals( +1, num10.getImaginary(), "The imaginary part should be 1 but it is "+num10.getImaginary()+"." );
		assertEquals( -1, num11.getImaginary(), "The imaginary part should be -1 but it is "+num11.getImaginary()+"." );
		
	}
	
	@Test
	public void testParseMethodWithTwoArgs() {
		ComplexNumber num12 = parse("-2.71-3.15i");
		ComplexNumber num13 = parse("31+24i");
		ComplexNumber num14 = parse("-1-i");
		ComplexNumber num15 = parse("+2.71+3.15i");
		assertEquals( -2.71, num12.getReal(), "The real part should be -2.71 but it is "+num12.getReal()+"." );
		assertEquals( -3.15, num12.getImaginary(), "The imaginary part should be -3.15 but it is "+num12.getImaginary()+"." );
		assertEquals( 31, num13.getReal(), "The real part should be 31 but it is "+num13.getReal()+"." );
		assertEquals( 24, num13.getImaginary(), "The imaginary part should be 24 but it is "+num13.getImaginary()+"." );
		assertEquals( -1, num14.getReal(), "The real part should be -1 but it is "+num14.getReal()+"." );
		assertEquals( -1, num14.getImaginary(), "The imaginary part should be -1 but it is "+num14.getImaginary()+"." );
		assertEquals( 2.71, num15.getReal(), "The real part should be 2.71 but it is "+num15.getReal()+"." );
		assertEquals( 3.15, num15.getImaginary(), "The imaginary part should be 3.15 but it is "+num15.getImaginary()+"." );
		assertThrows( NumberFormatException.class, () -> parse("-i3.17"), "The test should throw NumberFormatException but it doesn't." );
		assertThrows( NumberFormatException.class, () -> parse("i3.17"), "The test should throw NumberFormatException but it doesn't." );
		assertThrows( NumberFormatException.class, () -> parse("-+2.71"), "The test should throw NumberFormatException but it doesn't." );
		assertThrows( NumberFormatException.class, () -> parse("+2.71++3.15i"), "The test should throw NumberFormatException but it doesn't." );
	}
	
	@Test
	public void testParseMethodScientificNotation() {
		ComplexNumber num16 = parse("-3E-3+3E2i");
		ComplexNumber num17 = parse("4E+2+i");
		assertEquals( -0.003 , num16.getReal(), "The real part should be -0.003 but it is "+num16.getReal()+"." );
		assertEquals( 300, num16.getImaginary(), "The imaginary part should be 300 but it is "+num16.getImaginary()+"." );
		assertEquals( 400 , num17.getReal(), "The real part should be 400 but it is "+num16.getReal()+"." );
		assertEquals( 1, num17.getImaginary(), "The imaginary part should be 1 but it is "+num16.getImaginary()+"." );
	}
}
