package hr.fer.oprpp1.hw05.crypto.test;

import static hr.fer.oprpp1.hw05.crypto.Util.*;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

public class TestByteToHex {

	@Test
	public void testOnlySmallLetters() {

		String letters = "ffdfffdd";
		byte[] buff = {-1, -33, -1, -35};
		
		assertTrue( letters.equals(bytetohex(buff)) );
		
	}
	
	@Test
	public void testLettersAndNumbers() {

		String letters = "ffbaff02ff22";
		byte[] buff = {-1, -70, -1, 2, -1, 34};
		
		assertTrue( letters.equals(bytetohex(buff)) );
		
	}
	
	@Test
	public void testOnlyNumbers() {

		String letters = "1234567890";
		byte[] buff = {18, 52, 86, 120, -112};
		
		assertTrue( letters.equals(bytetohex(buff)) );
		
	}
	
	@Test
	public void testGivenWord() {

		String letters = "01ae22";
		byte[] buff = {1, -82, 34};
		
		assertTrue( letters.equals(bytetohex(buff)) );
	}
	
	@Test
	public void testDeadbeef() {

		String letters = "deadbeef";
		byte[] buff = {-34, -83, -66, -17};
		
		assertTrue( letters.equals(bytetohex(buff)) );
	}
	
	@Test
	public void testCafedude() {

		String letters = "cafed00d";
		byte[] buff = {-54, -2, -48, 13};
		
		assertTrue( letters.equals(bytetohex(buff)) );
	}
	
	@Test
	public void testDecafbad() {

		String letters = "decafbad";
		byte[] buff = {-34, -54, -5, -83};
		
		assertTrue( letters.equals(bytetohex(buff)) );
	}
	
	
}
