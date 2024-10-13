package hr.fer.zemris.java.hw05.shell;

import java.util.ArrayList;
import java.util.NoSuchElementException;
import java.util.stream.Stream;

/**
 * The method used for demonstrating the functionalities of the shell.
 * Reads input and executes the commands.
 * 
 * @author Ivan Rep
 */
public class MyShell {


	/**
	 * The main method. Called once the program starts.
	 * 
	 * @param args
	 * @throws ShellIOException if there is an error while reading the input.
	 */
	public static void main(String[] args) {

		Environment env = new ShellEnvironment();
		ShellStatus status = ShellStatus.CONTINUE;
		env.writeln("Welcome to MyShell v1.0");
		ArrayList<String> currCommand = new ArrayList<>();


		do {
			String line = null;

			env.write( env.getPromptSymbol()+" ");

			try { line=env.readLine(); } 
			catch(IllegalStateException | NoSuchElementException exc) { 
				env.writeln(new ShellIOException("An error occured while reading the input. Try again.").getMessage());
				}

			line = line.trim();
			String s[] = line.split("\\s+");

			if( s[s.length-1].equals(String.valueOf(env.getMorelinesSymbol())) ) {
				keepReading(s, env, currCommand);
				
				StringBuilder sb = new StringBuilder();
				currCommand.forEach( str -> sb.append(str+" ") );
				line = sb.toString();
				line = line.trim();
				currCommand.clear();
				
			}


			String commandName = line.substring( 0, line.indexOf(' ')==-1 ? line.length() : line.indexOf(' ') );
			ShellCommand command = env.commands().get(commandName);
			try {
				status = command.executeCommand(env, 
						line.indexOf(' ')==-1 ? "" : line.substring( line.indexOf(' '), line.length() ));
			} catch(IllegalStateException | NoSuchElementException exc) {
				env.writeln("An error occured while executing a command. Try again.");
			}



		} while(!status.equals(ShellStatus.TERMINATE));


	}

	/** 
	 * Method used for reading input when the last symbol in the input is 
	 * a morelines symbol.
	 * 
	 * @param s an array of string from the last read input
	 * @param env the enviroment of the shell
	 * @param currCommand a list of strings for the complete command input
	 */
	private static void keepReading(String[] s, Environment env, ArrayList<String> currCommand) {
		while( s[s.length-1].equals(String.valueOf( env.getMorelinesSymbol() ) ) ) {
			Stream.of(s).forEach( str -> currCommand.add(str) );
			currCommand.remove(currCommand.size()-1);
			
			String line = null;
			env.write( String.valueOf( env.getMultilineSymbol()+" " ) );
			line = env.readLine();
			line = line.trim();
			s = line.split("\\s+");
		}
		
		Stream.of(s).forEach( str -> currCommand.add(str) );
	}
	

}
