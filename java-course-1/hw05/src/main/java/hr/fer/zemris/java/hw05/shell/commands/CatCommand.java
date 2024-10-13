package hr.fer.zemris.java.hw05.shell.commands;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import hr.fer.zemris.java.hw05.shell.Environment;
import hr.fer.zemris.java.hw05.shell.ShellCommand;
import hr.fer.zemris.java.hw05.shell.ShellStatus;

/**
 * Class represents a command for reading a file
 * and printing it on the console.
 * 
 * @author Ivan Rep
 */
public class CatCommand implements ShellCommand, QuotationSkipper {

	private static final Integer LENGTH_4096 = 4096;
	private static final String commandName = "cat";
	private static final String[] description = 
		{"The command prints all available charsets.",
		"Syntax: 'cat path_to_file charset' "};


	/**
	 * The method reads the content of the given file and prints it on the console.
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

		Map<String, Charset> map = Charset.availableCharsets();	

		if(args.length>2) {
			env.writeln("Invalid amount of arguments for charset command.");
			return ShellStatus.CONTINUE;
		} else if (args.length==2 && !map.containsKey(args[1])) {
			env.writeln("The given charset doesn't exist or isn't supported.");
			return ShellStatus.CONTINUE;
		} else if (!f.exists()) {
			env.writeln("The given file doesn't exist.");
			return ShellStatus.CONTINUE;
		} else if (!f.canRead()) {
			env.writeln("The given file can't be read.");
			return ShellStatus.CONTINUE;
		}

		Charset c =  args.length==2 ? Charset.forName(args[1]) : Charset.defaultCharset();
		Path p = Paths.get(args[0]);
		StringBuilder sb = new StringBuilder();
		
//		try (InputStream is = Files.newInputStream(p)) {
//			byte[] buff = new byte[LENGTH_4096];
//			// do sth before {}
//			while(true) {
//				int r = is.read(buff);
//				if(r<1) break;
//				// do smth while {
//				sb.append( new String(buff, c) );
//				// }
//			}
//			// do smth after {
//			env.writeln(sb.toString()); 
//			// }
//			is.close();
//		} catch(IOException exc) {
//			env.writeln(exc.getMessage());
//			env.writeln("Something went wrong while reading or writing the files.");
//		}

		byte[] buff = new byte[LENGTH_4096];
		
		DataReader copyReader = new DataReader() {

			private StringBuilder sb;

			@Override
			public void doBefore() { sb = new StringBuilder(); }

			@Override
			public void doWhile(int r) { sb.append( new String(buff, c) ); }

			@Override
			public void doAfter() { env.writeln(sb.toString()); }

		};

		copyReader.readAndDo(env, p, buff);
		
		return ShellStatus.CONTINUE;
	}

	/**
	 * Getter method for the cat command's name;
	 * 
	 * @return the string representation of the command name
	 */
	@Override
	public String getCommandName() {
		return commandName;
	}

	/**
	 * Retrieves the cat command's description.
	 * 
	 * @return the list containing the command description
	 */
	@Override
	public List<String> getCommandDescription() {
		return Collections.unmodifiableList( Arrays.asList(description) );
	}

}
