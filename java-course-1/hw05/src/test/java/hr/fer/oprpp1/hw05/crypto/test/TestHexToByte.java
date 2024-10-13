package hr.fer.oprpp1.hw05.crypto.test;

import org.junit.jupiter.api.Test;
import static hr.fer.oprpp1.hw05.crypto.Util.*;
import static org.junit.jupiter.api.Assertions.*;


public class TestHexToByte {

	@Test
	public void testOnlySmallLetters() {

		String letters = "ffdfffdd";
		byte[] buff = {-1, -33, -1, -35};
		
		assertArrayEquals(buff, hextobyte(letters) );
		
	}
	
	@Test
	public void testOnlyBigLetters() {

		String letters = "FFDFFFDD";
		byte[] buff = {-1, -33, -1, -35};
		
		assertArrayEquals(buff, hextobyte(letters) );
		
	}
	
	@Test
	public void testOnlyBigAndSmallLetters() {

		String letters = "FfDffFDd";
		byte[] buff = {-1, -33, -1, -35};
		
		assertArrayEquals(buff, hextobyte(letters) );
		
	}
	
	@Test
	public void testLettersAndNumbers() {

		String letters = "ffbaff02ff22";
		byte[] buff = {-1, -70, -1, 2, -1, 34};
		
		assertArrayEquals(buff, hextobyte(letters) );
		
	}
	
	@Test
	public void testOnlyNumbers() {

		String letters = "1234567890";
		byte[] buff = {18, 52, 86, 120, -112};
		
		assertArrayEquals(buff, hextobyte(letters) );
		
	}
	
	@Test
	public void testGivenWord() {

		String letters = "01Ae22";
		byte[] buff = {1, -82, 34};
		
		assertArrayEquals(buff, hextobyte(letters) );
		
	}
	
	@Test
	public void testDeadbeef() {

		String letters = "deadbeef";
		byte[] buff = {-34, -83, -66, -17};
		
		assertArrayEquals(buff, hextobyte(letters) );
		
	}
	
}
