package hr.fer.zemris.java.hw05.shell.commands;

import static hr.fer.oprpp1.hw05.crypto.Util.*;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import hr.fer.zemris.java.hw05.shell.*;

/**
 * Class for the command that prints the hex content of the
 * given file.
 * 
 * @author Ivan Rep
 */
public class HexDumpCommand implements ShellCommand, QuotationSkipper {

	private static final String commandName = "hexdump";
	private static final String[] description = 
		{"The command produces hex output of the given file.",
		"Syntax: 'hexdump path_to_file' "};
	private static final Integer LENGTH_16 = 16;

	/**
	 * Method prints the hex content of the given file.
	 * If there is a problem with execution a message is printed to the console.
	 * 
	 * @param env the environment of the shell
	 * @param arguments the string containing everything after the key word
	 * @return a shell status enum
	 */
	@Override
	public ShellStatus executeCommand(Environment env, String arguments) {

//		String[] args = arguments.trim().split("\\s+");
		
		if(arguments.isEmpty()) {
			env.writeln("The given input is empty.");
			return ShellStatus.CONTINUE;
		}
		
		String[] args = QuotationSkipper.skipQuotesAndSplit(arguments);
		if(args == null) {
			env.writeln("The given input has a problem with escaping characters or has too many quotation marks. Try again.");
			return ShellStatus.CONTINUE;
		}
		
		File f = new File(args[0]);

		if(arguments.isEmpty() || args.length!=1) {
			env.writeln("The command takes in 1 argument.");
			return ShellStatus.CONTINUE;
		} else if(f.isDirectory()) {
			env.writeln("The given path should be to a file, not to a directory.");
			return ShellStatus.CONTINUE;
		} else if(!f.canRead()) {
			env.writeln("The given file can't be read.");
			return ShellStatus.CONTINUE;
		}

//		int count = 0;
//		String hex = "";
//		try (InputStream is = Files.newInputStream(f.toPath().toAbsolutePath()) ) {
//			byte[] buff = new byte[LENGTH_16]; 
//			// do smth before {}
//			while(true) {
//				int r = is.read(buff);
//				if(r<1) break;
//				// do smth while {
//				if(r == LENGTH_16) format(new String(buff), hex, buff, count, env); 
//				else format(new String(buff).substring(0, r), hex.substring(0, r), buff, count, env);
//				count+=LENGTH_16;
//				// }
//			}
//			// do smth after { }
//			is.close(); 
//		} catch(IOException exc) {
//			env.writeln(exc.getMessage());
//			env.writeln("Something went wrong while reading or writing the files.");
//		}

		byte[] buff = new byte[LENGTH_16]; 
		
		DataReader hexDumpReader = new DataReader() {
			
			private int count = 0;
			
			@Override
			public void doWhile(int r) {
				String hex = bytetohex(buff); 
				if(r == LENGTH_16) format(new String(buff), hex, buff, count, env); 
				else format(new String(buff).substring(0, r), hex.substring(0, r), buff, count, env);
				count+=LENGTH_16;
			}

		};
		
		hexDumpReader.readAndDo(env, f.toPath().toAbsolutePath(), buff);


		return ShellStatus.CONTINUE;	
	}

	/**
	 * Method formats the output for the hexdump.
	 * Prints a hex formatted integer, the hexes and the content 
	 * in the file.
	 * 
	 * @param text the string representation of the text
	 * @param hex the hex representation of the text
	 * @param buff the array of bytes correspoding to the hex
	 * @param count the count of the hex values already written
	 * @param env the shell environment
	 */
	private void format(String text, String hex, byte[] buff, int count, Environment env) {

		String[] s = turnIntoArraysOfTwo(buff, hex);

		env.write(String.format("%08X:", count));

		for(int i=0; i<LENGTH_16/2; i++) 
			if(i<s.length) env.write(" "+s[i].toUpperCase());
			else env.write("   ");
		
		env.write("|");

		for(int i=LENGTH_16/2; i<LENGTH_16; i++) {
			if(i<s.length) env.write( s[i].toUpperCase() );
			else env.write("  " );
			env.write( i!=LENGTH_16-1 ? " " : "" );
		}
		env.write("| ");

		StringBuilder sb = new StringBuilder();
		text.chars()
		.forEach( c -> { if( c>=32 && c<=127 ) sb.append(new String(String.valueOf((char)c)));
		else sb.append(".");
		});

		env.writeln(sb.toString());

	}

	/**
	 * The method calculates the string hex representation of the given byte array.
	 * Removes invalid characters.
	 * 
	 * @param buff the given byte array
	 * @param hex the current hex
	 * @return an array of hexes
	 */
	private String[] turnIntoArraysOfTwo(byte[] buff, String hex) {

		int offset = 2;
		String[] s = new String[hex.length()/offset];

		for(int i=0, j=0; j<hex.length()/offset; i+=offset, j++) {
			s[j] = hex.substring(i, i+offset);
			if(  buff[j]<32 || buff[j]>127  ) {
				buff[j] = (byte)'.';
			}
		}

		return s;
	}


	/**
	 * Getter method for the hexdump command's name;
	 * 
	 * @return the string representation of the command name
	 */
	@Override
	public String getCommandName() {
		return commandName;
	}

	/**
	 * Retrieves the hexdump command's description.
	 * 
	 * @return the list containing the command description
	 */
	@Override
	public List<String> getCommandDescription() {
		return Collections.unmodifiableList( Arrays.asList(description) );
	}


}
