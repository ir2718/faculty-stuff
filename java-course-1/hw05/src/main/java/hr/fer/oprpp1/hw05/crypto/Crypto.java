package hr.fer.oprpp1.hw05.crypto;

import static hr.fer.oprpp1.hw05.crypto.Util.hextobyte;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.*;
import java.security.*;
import java.security.spec.AlgorithmParameterSpec;
import java.util.Scanner;
import javax.crypto.*;
import javax.crypto.spec.*;
import static hr.fer.oprpp1.hw05.crypto.Util.*;

/**
 * The class is used for decrypting and encrypting 
 * different kinds of files and calculating the message
 * digest of a file.
 * 
 * @author Ivan Rep
 */
public class Crypto {

	private static final Integer LENGTH_4096 = 4096;

	/**
	 * The main method
	 * 
	 * @param args The command, either 'checksha', 'encrypt' or 'decrypt'
	 * @throws NoSuchAlgorithmException if the given algorithm is not valid, sha-256 is used here
	 * @throws NoSuchPaddingException if a particular padding mechanism is requested but is not available in the environment
	 * @throws InvalidKeyException if the key is invalid
	 * @throws InvalidAlgorithmParameterException if the exception for invalid or inappropriate algorithm parameters
	 * @throws FileNotFoundException if an attempt to open the file denoted by a specified pathname has failed. 
	 * @throws IllegalBlockSizeException if the length of data provided to a block cipher is incorrect
	 * @throws BadPaddingException if particular padding mechanism is expected for the input data but the data is not padded properly
	 * @throws ShortBufferException if an output buffer provided by the user is too short to hold the operation result
	 */
	public static void main(String[] args) throws NoSuchAlgorithmException, NoSuchPaddingException, InvalidKeyException, InvalidAlgorithmParameterException, FileNotFoundException, IllegalBlockSizeException, BadPaddingException, ShortBufferException {

		Scanner scan = new Scanner(System.in);
		String url = args[1];

		if ( args[0].equals("checksha") && args.length==2 ){

			System.out.print("Please provide expected sha-256 digest for "+url+":\n> ");

			String expected = scan.nextLine();
			
			InputStream is = null;
			MessageDigest digest = MessageDigest.getInstance("SHA-256");
			byte[] digestArr = null;

			try {
				is = Files.newInputStream( Paths.get(url) );
				byte[] buff = new byte[LENGTH_4096];
				int offset = 0;

				while(true) {
					int r = is.read(buff);
					if(r<1) break;
					digest.update(buff, offset, r);
				}

				digestArr = digest.digest();

			} catch (IOException ex) {	
				System.out.println("IOException happened.");
			} finally {
				if(is!=null)
					try { is.close(); }
				catch(IOException ignorable) { }
			}

			String s = bytetohex(digestArr);
			if( s.equals(expected) ) System.out.println("Digesting completed. Digest of "+args[1]+" matches expected digest.");
			else System.out.println("Digesting completed. Digest of "+args[1]+" does not match the expected digest. Digest "
					+ "was: "+s);


		} else if( (args[0].equals("encrypt") && args.length==3) || (args[0].equals("decrypt") && args.length==3) ) {

			System.out.print("Please provide password as hex-encoded text (16 bytes, i.e. 32 hex-digits):\n> ");
			String password = scan.nextLine();

			System.out.print("Please provide initialization vector as hex-encoded text (32 hex-digits):\n> ");
			String initVector = scan.nextLine();

			Path p = Paths.get(args[2]);
			
			Cipher cipher = null;
			if( args[0].equals("encrypt") ) cipher = initialize(true, password, initVector);
			else if( args[0].equals("decrypt") ) cipher = initialize(false, password, initVector);

			InputStream is = null;
			OutputStream os = null;
			
			try {	
				is = Files.newInputStream(Paths.get(url));
				os = Files.newOutputStream(p);
				byte[] buff = new byte[LENGTH_4096];
				int offset=0;

				while(true) {
					int r = is.read(buff);
					if(r<1) break;
					os.write( cipher.update(buff, offset, r) );
				}
				
				os.write( cipher.doFinal() );

			} catch (IOException ex) {
				System.out.println("IOException happened.");
			} finally {
				if(is!=null)
					try { is.close();
						  os.close(); }
				catch(IOException ignorable) { }
			}

			System.out.println("Encryption completed. Generated file "+args[2]+" based on file "+args[1]+".");

		}  else {
			scan.close();
			throw new IllegalArgumentException("The input isn't correct. Try again.");
		}

		scan.close();

	}
	
	/**
	 * 
	 * @param encrypt true if used for encrypting, false if used for decrypting
	 * @param password the user's password read as input
	 * @param initVector the user's initialization vector read as input
	 * @return the cipher initialized from the given values
	 * @throws NoSuchAlgorithmException if the given algorithm is not valid, sha-256 is used here
	 * @throws NoSuchPaddingException if a particular padding mechanism is requested but is not available in the environment
	 * @throws InvalidKeyException if the key is invalid
	 * @throws InvalidAlgorithmParameterException if the exception for invalid or inappropriate algorithm parameters
	 */
	private static Cipher initialize(boolean encrypt, String password, String initVector) throws NoSuchAlgorithmException, NoSuchPaddingException, InvalidKeyException, InvalidAlgorithmParameterException {
		String keyText = password;
		String ivText = initVector;
		SecretKeySpec keySpec = new SecretKeySpec(Util.hextobyte(keyText), "AES");
		AlgorithmParameterSpec paramSpec = new IvParameterSpec(hextobyte(ivText));
		Cipher cipher = Cipher.getInstance("AES/CBC/PKCS5Padding");
		cipher.init(encrypt ? Cipher.ENCRYPT_MODE : Cipher.DECRYPT_MODE, keySpec, paramSpec);
		return cipher;
	}

}
