package hr.fer.oprpp1.hw05.crypto;

import java.util.Arrays;

/**
 * The class is used for transforming hex values to 
 * byte values and vice versa.
 * 
 * @author Ivan Rep
 */
public class Util {

	public static final Integer ZERO_ASCII_VALUE = 48;
	public static final Integer TEN_TO_HEX_A = 55;
	public static final Integer LENGTH_CONSTANT = 4;
	public static final Integer HEXES_IN_BYTE = 2;
	public static final Integer BASE = 2;
	public static final Integer MASK_F = 0xF;

	/**
	 * Returns a byte array for the given text.
	 * 
	 * @param keyText the given text
	 * @return the corresponding byte array
	 */
	public static byte[] hextobyte(String keyText) {
		
		byte[] b = new byte[keyText.length()/HEXES_IN_BYTE];
		char[] charArr = keyText.toCharArray();
		int[] num = new int[BASE];
		
		for(int i=0, j=0; i<keyText.length(); i+=BASE, j++) {
			for(int k=0; k<BASE; k++) 
				if( Character.isDigit( charArr[i+k] ) )
					num[k] = ( ( (byte) charArr[i+k] - ZERO_ASCII_VALUE ) ) << LENGTH_CONSTANT*(BASE-1-k);
				else
					num[k] = ( ( (byte) Character.toUpperCase( charArr[i+k] ) - TEN_TO_HEX_A ) ) << LENGTH_CONSTANT*(BASE-1-k);
			
			b[j] = (byte) Arrays.stream(num).sum();
		}
	
		return b;
	}

	
	/**
	 * Returns a hex for the given byte array.
	 * 
	 * @param keyText the given byte array
	 * @return the corresponding hex string
	 */
	public static String bytetohex(byte[] keyText) {
		
		StringBuilder sb = new StringBuilder();
		for(int i=0; i<keyText.length; i++) 
			sb.append( findString( (keyText[i]>>>LENGTH_CONSTANT) & MASK_F ) ).append( findString( keyText[i] & MASK_F ) );
		
		return sb.toString();
	}


	/**
	 * Returns the hex value of an integer.
	 * For values from 1 to 10 returns the given number.
	 * From 10 to 16 returns a letter of the alphabet.
	 * 
	 * @param i the given integer
	 * @return hex value of the given integer
	 */
	private static String findString(int i) {
		
		if(i<10)
			return String.valueOf(i);
		
		return String.valueOf( Character.toLowerCase( (char) (i + TEN_TO_HEX_A) ) );
		
	}


}
