package hr.fer.zemris.java.hw05.shell.commands;

import java.io.IOException;
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
 * Class for the command that makes a directory with the given path.
 * 
 * @author Ivan Rep
 */
public class MkDirCommand implements ShellCommand, QuotationSkipper {

	private static final String commandName = "mkdir";
	private static final String[] description = 
		{"The command creates a new directory with the given path.",
		"If such already exists, a message is printed to the user.",
		"Syntax: 'mkdir path' "};
	
	/**
	 * Method creates a directory with the given path.
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
		
		if(args.length!=1) {
			env.writeln("The method takes in 1 argument.");
			return ShellStatus.CONTINUE;
		}
		
        try {

            Path path = Paths.get(args[0]);
            
            if( path.toFile().exists() && path.toFile().isDirectory() ) {
            	env.writeln("The given directory already exists and can't be created.");
            	return ShellStatus.CONTINUE;
            }

            Files.createDirectories(path);
            env.writeln("Directory with given path was created.");
            
        } catch (IOException e) {
            env.writeln("Something went wrong while creating a new directory. Try again.");
        }
        
		return ShellStatus.CONTINUE;
	}

	/**
	 * Getter method for the mkdir command's name;
	 * 
	 * @return the string representation of the command name
	 */
	@Override
	public String getCommandName() {
		return commandName;
	}

	/**
	 * Retrieves the mkdir command's description.
	 * 
	 * @return the list containing the command description
	 */
	@Override
	public List<String> getCommandDescription() {
		return Collections.unmodifiableList( Arrays.asList(description) );
	}
}
