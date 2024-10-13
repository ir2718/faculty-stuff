package hr.fer.zemris.java.hw05.shell;

import java.util.List;

/**
 * Representation of a command in the shell.
 * 
 * @author Ivan Rep
 */
public interface ShellCommand {

	/**
	 * Executes the command and returns the status of the shell.
	 * 
	 * @param env the environment of the shell
	 * @param arguments the string representation of the given arguments after the key word
	 * @return the status of the shell
	 */
	ShellStatus executeCommand(Environment env, String arguments);
	
	/**
	 * Getter method for the command name.
	 * 
	 * @return the string representation of the command name
	 */
	String getCommandName();
	
	/**
	 * A list of sentences describing the functionalities of the command
	 * and the syntax.
	 * 
	 * @return the description in the form of a list of strings
	 */
	List<String> getCommandDescription();

}
