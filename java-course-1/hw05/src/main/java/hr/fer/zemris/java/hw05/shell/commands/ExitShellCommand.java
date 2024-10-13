package hr.fer.zemris.java.hw05.shell.commands;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import hr.fer.zemris.java.hw05.shell.Environment;
import hr.fer.zemris.java.hw05.shell.ShellCommand;
import hr.fer.zemris.java.hw05.shell.ShellEnvironment;
import hr.fer.zemris.java.hw05.shell.ShellStatus;

/**
 * Class for the command that shuts down the shell.
 * 
 * @author Ivan Rep
 */
public class ExitShellCommand implements ShellCommand {

	private static final String commandName = "exit";
	private static final String[] description = 
		{"The command exits the shell.",
		"Syntax: 'exit' "};
	
	/**
	 * Method shuts down the shell.
	 * If there is a problem with execution a message is printed to the console.
	 * 
	 * @param env the environment of the shell
	 * @param arguments the string containing everything after the key word
	 * @return a shell status enum
	 */
	@Override
	public ShellStatus executeCommand(Environment env, String arguments) {

		if(!arguments.isEmpty()) {
			env.writeln("The command expects no arguments after 'exit'.");
			return ShellStatus.CONTINUE;
		}
		
		env.writeln("Shutting down MyShell...");
		((ShellEnvironment)env).exit();
		return ShellStatus.TERMINATE;
	}

	/**
	 * Getter method for the exit shell command's name;
	 * 
	 * @return the string representation of the command name
	 */
	@Override
	public String getCommandName() {
		return commandName;
	}

	/**
	 * Retrieves the exit shell command's description.
	 * 
	 * @return the list containing the command description
	 */
	@Override
	public List<String> getCommandDescription() {
		return Collections.unmodifiableList( Arrays.asList(description) );
	}
	

}
