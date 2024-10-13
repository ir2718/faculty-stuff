package hr.fer.zemris.java.hw05.shell.commands;

import static hr.fer.oprpp1.hw05.crypto.Util.bytetohex;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import hr.fer.zemris.java.hw05.shell.Environment;
import hr.fer.zemris.java.hw05.shell.ShellCommand;
import hr.fer.zemris.java.hw05.shell.ShellStatus;

/**
 * Class for the command that copies the content of one file to the other.
 * 
 * @author Ivan Rep
 */
public class CopyCommand implements ShellCommand, QuotationSkipper {

	private static final Integer LENGTH_4096 = 4096;
	private static final String commandName = "copy";
	private static final String[] description = 
		{"The command copies the content of the first file into the second file.",
				"The second argument is optional.",
		"Syntax: 'copy file_1 file_2' "};
	private static boolean caught = false;

	/**
	 * Method prints the copies the content of one file to the other.
	 * The paths of the files are given in the arguments value.
	 * If there is a problem with execution a message is printed to the console.
	 * 
	 * @param env the environment of the shell
	 * @param arguments the string containing everything after the key word
	 * @return a shell status enum
	 */
	@Override
	public ShellStatus executeCommand(Environment env, String arguments) {

		//		String[] args = arguments.trim().split("\\s+");
		String[] args = QuotationSkipper.skipQuotesAndSplit(arguments);
		if(args == null) {
			env.writeln("The given input has a problem with escaping characters or has too many quotation marks. Try again.");
			return ShellStatus.CONTINUE;
		}

		File f1, f2;
		if( args.length==2 ) {
			f1 = new File(args[0]);
			f2 = new File(args[1]);
		} else {
			env.writeln("The command takes in 2 arguments.");
			return ShellStatus.CONTINUE;
		}

		if(!f1.exists()) {
			env.writeln("The first file doesn't exist.");
			return ShellStatus.CONTINUE;
		} else if (f1.isDirectory()) {
			env.writeln("The first file is a directory, but it must be a file.");
			return ShellStatus.CONTINUE;
		} else if (!f1.canRead()) {
			env.writeln("The first file can't be read.");
			return ShellStatus.CONTINUE;
		}


		// f2 is a file
		if( !f2.exists() ) {
			readAndWrite(f1, f2, env);
			if(caught) {
				caught = false;
				return ShellStatus.CONTINUE;
			}
			env.writeln("A new file with the given name and contents was created in the same directory.");
			return ShellStatus.CONTINUE;
		}

		if(f2.exists() && f2.isFile() ) {

			if(overWrite(env)) {
				readAndWrite(f1, f2, env);
				if(caught) {
					caught = false;
					return ShellStatus.CONTINUE;
				}
				env.writeln("The file was overwritten.");
			}

			return ShellStatus.CONTINUE;
		}


		// f2 is a directory
		Path newPath = Paths.get(f2.toPath().toAbsolutePath()
				+"\\"+ f1.toPath().getFileName());
		File newFile = newPath.toFile();

		if(newFile.exists()) {

			if(overWrite(env)) {
				readAndWrite(f1, newFile, env);
				if(caught) {
					caught = false;
					return ShellStatus.CONTINUE;
				}
				env.writeln("The file was overwritten in the given directory.");
			}

			return ShellStatus.CONTINUE;
		} 

		readAndWrite(f1, newFile, env);
		env.writeln("The file was copied and created in the given directory.");
		return ShellStatus.CONTINUE;

	}


	/**
	 * Getter method for the copy command's name;
	 * 
	 * @return the string representation of the command name
	 */
	@Override
	public String getCommandName() {
		return commandName;
	}

	/**
	 * Retrieves the copy command's description.
	 * 
	 * @return the list containing the command description
	 */
	@Override
	public List<String> getCommandDescription() {
		return Collections.unmodifiableList( Arrays.asList(description) );
	}


	/**
	 * A method used for reading the file f1 and writing into the file f2.
	 * 
	 * @param f1 the file that is read
	 * @param f2 the file that is written into
	 * @param env the shell environment
	 */
	private static void readAndWrite(File f1, File f2, Environment env) {

		//		try (InputStream is = Files.newInputStream(f1.toPath())) {
		//			byte[] buff = new byte[LENGTH_4096];
		//			// do smth before {
		//			OutputStream os = Files.newOutputStream(f2.toPath());
		//			// }
		//			while(true) {
		//				int r = is.read(buff);
		//				if(r<1) break;
		//				// do smth while {
		//				os.write(buff);
		//				// }
		//			}
		//			// do smth after {
		//			os.close();
		//			// }
		//			is.close();
		//		} catch(IOException exc) {
		//			caught = true;
		//			env.writeln(exc.getMessage());
		//			env.writeln("Something went wrong while reading or writing the files.");
		//		}


		byte[] buff = new byte[LENGTH_4096]; 

		DataReader copyReader = new DataReader() {

			private OutputStream os;

			@Override
			public void doBefore() {
				try { os = Files.newOutputStream(f2.toPath()); }
				catch (IOException e) { }
			}

			@Override
			public void doWhile(int r) {
				try { os.write(buff); }
				catch (IOException e) { }
			}

			@Override
			public void doAfter() {
				try { os.close(); } 
				catch (IOException e) { }
			}

		};

		copyReader.readAndDo(env, f1.toPath(), buff);

	}

	/**
	 * Method overwrites checks if the user agrees to overwriting file.
	 * 
	 * @param env the shell environment
	 * @return true if user agrees, otherwise false
	 */
	private static boolean overWrite(Environment env) {

		env.writeln("The file already exists. Do you want to overwrite it?"
				+ "\n Write 'yes' or 'no'.");

		String res = env.readLine();
		if(res.trim().toUpperCase().equals("NO")) {
			env.writeln("Copying stopped.");
			return false;
		} else if ( !res.trim().toUpperCase().equals("YES") ) {
			env.writeln("The given response isn't valid. Copying stopped.");
			return false;
		}

		return true;
	}


}
