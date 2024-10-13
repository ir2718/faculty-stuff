package hr.fer.zemris.java.hw05.shell.commands;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.SortedMap;

import hr.fer.zemris.java.hw05.shell.Environment;
import hr.fer.zemris.java.hw05.shell.ShellCommand;
import hr.fer.zemris.java.hw05.shell.ShellStatus;

/**
 * Class for the command that prints descriptions of the 
 * given command or the names of all the commands.
 * 
 * @author Ivan Rep
 */
public class HelpCommand implements ShellCommand {

	private static final String commandName = "help";
	private static final String[] description = 
		{"If started with no arguments, lists names of all supported commands.",
				"If started with single argument, prints name and the description of selected command.",
		"Syntax: 'help' or 'help command_name' "};

	
	/**
	 * Method prints descriptions of the given command 
	 * or the names of all the commands.
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
		
		SortedMap<String, ShellCommand> map = env.commands();
		
		if(arguments.isEmpty()) {
	
			env.writeln("The supported commands are:");
			map.keySet().stream().forEach( name -> env.writeln(name) );
			return ShellStatus.CONTINUE;

		} else if (args.length!=1) 
			return ShellStatus.CONTINUE;

		ShellCommand sc = map.get(args[0]);
		sc.getCommandDescription().stream().forEach( line -> env.writeln(line) );

		return ShellStatus.CONTINUE;
	}

	/**
	 * Getter method for the help command's name;
	 * 
	 * @return the string representation of the command name
	 */
	@Override
	public String getCommandName() {
		return commandName;
	}

	
	/**
	 * Retrieves the help command's description.
	 * 
	 * @return the list containing the command description
	 */
	@Override
	public List<String> getCommandDescription() {
		return Collections.unmodifiableList( Arrays.asList(description) );
	}



}
