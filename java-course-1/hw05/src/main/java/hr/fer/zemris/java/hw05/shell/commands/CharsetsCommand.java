package hr.fer.zemris.java.hw05.shell.commands;

import java.nio.charset.Charset;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import hr.fer.zemris.java.hw05.shell.Environment;
import hr.fer.zemris.java.hw05.shell.ShellCommand;
import hr.fer.zemris.java.hw05.shell.ShellStatus;

/**
 * Class for the command that prints the available charsets to the console.
 * 
 * @author Ivan Rep
 */
public class CharsetsCommand implements ShellCommand {
	
	private static final String commandName = "charsets";
	private static final String[] description = 
		{"The command prints all available charsets.",
		"Syntax: 'charsets'"};

	/**
	 * Method prints the names of the available charsets.
	 * If there is a problem with execution a message is printed to the console.
	 * 
	 * @param env the environment of the shell
	 * @param arguments the string containing everything after the key word
	 * @return a shell status enum
	 */
	@Override
	public ShellStatus executeCommand(Environment env, String arguments) {

		if(!arguments.isEmpty()) {
			env.writeln("Charsets input format is wrong. Type in 'charsets' for available charsets.");
			return ShellStatus.CONTINUE;
		}
				
		Map<String, Charset> map = Charset.availableCharsets();
		
		for( Map.Entry<String, Charset> entry : map.entrySet()  ) {
			env.writeln(entry.getKey()+" -> "+entry.getValue());;
		}
		
		return ShellStatus.CONTINUE;
	}

	/**
	 * Getter method for the charsets command's name;
	 * 
	 * @return the string representation of the command name
	 */
	@Override
	public String getCommandName() {
		return commandName;
	}

	/**
	 * Retrieves the charsets command's description.
	 * 
	 * @return the list containing the command description
	 */
	@Override
	public List<String> getCommandDescription() {
		return Collections.unmodifiableList( Arrays.asList(description) );
	}

}
