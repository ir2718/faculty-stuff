package hr.fer.zemris.java.hw05.shell.commands;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import hr.fer.zemris.java.hw05.shell.Environment;
import hr.fer.zemris.java.hw05.shell.ShellCommand;
import hr.fer.zemris.java.hw05.shell.ShellStatus;

/**
 * Class for the command regarding symbols. 
 * Gets and sets the prompt, mutiline and morelines symbol.
 * 
 * @author Ivan Rep
 */
public class SymbolCommand implements ShellCommand {

	private static final String PROMPT = "PROMPT";
	private static final String MORELINES = "MORELINES";
	private static final String MULTILINE = "MULTILINE";

	private static final String commandName = "symbol";
	private static final String[] description = 
		{"The command is used for setting symbols and printing them.",
				"Syntax: 'symbol PROMPT' ",
				"Syntax: 'symbol PROMPT some_char' ",
				"Syntax: 'symbol MORELINES' ",
				"Syntax: 'symbol MORELINES some_char' ",
				"Syntax: 'symbol MULTILINES' ",
		"Syntax: 'symbol MULTILINES some_char' "};

	/**
	 * Method sets or gets the value of one of the current symbols.
	 * If there is a problem with execution a message is printed to the console.
	 * 
	 * @param env the environment of the shell
	 * @param arguments the string containing everything after the key word
	 * @return a shell status enum
	 */
	@Override
	public ShellStatus executeCommand(Environment env, String arguments) {
		
		if(arguments.isEmpty()) {
			env.writeln("The given input is empty.");
			return ShellStatus.CONTINUE;
		}

		String[] args = (arguments.trim()).split("\\s+");
		SymbolInterface command=null;

		if( args[0].equals(PROMPT) ) {
			command = new SymbolInterface() {
				
				@Override
				public Character getMethod() { return env.getPromptSymbol(); }
				
				@Override
				public void setMethod(char c) { env.setPromptSymbol(c); }
				
			};
			command.findSymbolCommand(args, env);
		} else if ( args[0].equals(MORELINES) ) {
			command = new SymbolInterface() {
				
				@Override
				public Character getMethod() { return env.getMorelinesSymbol(); }
				
				@Override
				public void setMethod(char c) { env.setMorelinesSymbol( c ); }
				
			};
			command.findSymbolCommand(args, env);
		} else if ( args[0].equals(MULTILINE) ) {
			command = new SymbolInterface() {
				
				@Override
				public Character getMethod() { return env.getMultilineSymbol(); }
				
				@Override
				public void setMethod(char c) { env.setMultilineSymbol( c ); }
				
			};
			command.findSymbolCommand(args, env);
		} else {
			env.writeln("The given symbol command isn't supported or doesn't exist.");
		}

		return ShellStatus.CONTINUE;
	}

	/**
	 * Getter method for the symbol command's name;
	 * 
	 * @return the string representation of the command name
	 */
	@Override
	public String getCommandName() {
		return commandName;
	}

	/**
	 * Retrieves the symbol command's description.
	 * 
	 * @return the list containing the command description
	 */
	@Override
	public List<String> getCommandDescription() {
		return Collections.unmodifiableList( Arrays.asList(description) );
	}


}
