package hr.fer.zemris.java.hw05.shell.commands;

import java.io.File;
import java.io.IOException;
import java.nio.file.*;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import hr.fer.zemris.java.hw05.shell.Environment;
import hr.fer.zemris.java.hw05.shell.ShellCommand;
import hr.fer.zemris.java.hw05.shell.ShellStatus;

/**
 * Class for the command that prints a tree of the files and directories.
 * 
 * @author Ivan Rep
 */
public class TreeCommand implements ShellCommand, QuotationSkipper {

	private static final String commandName = "tree";
	private static final String[] description = 
		{"The command prints all directories and files.",
		"Syntax: 'tree directory_name' "};
	
	
	/**
	 * Method prints the tree of files and directories.
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
		
		File f = new File(args[0]);
		
		if(args.length!=1) {
			env.writeln("The tree command takes only one directory.");
			return ShellStatus.CONTINUE;
		} else if (!f.isDirectory()) {
			env.writeln("The given file isn't a directory.");
			return ShellStatus.CONTINUE;
		}
		
		SimpleFileVisitor<Path> visitor = new SimpleFileVisitor<Path>() {
			int count = 0;
			
			@Override
			public FileVisitResult postVisitDirectory(Path dir, IOException exc) throws IOException {
				count-=2;
				return FileVisitResult.CONTINUE;
			}
			
			@Override
			public FileVisitResult preVisitDirectory(Path dir, BasicFileAttributes attrs) throws IOException {
				env.writeln(" ".repeat(count)+"[DIR] "+dir.getFileName());
				count+=2;
				return FileVisitResult.CONTINUE;
			}
			
			@Override
			public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) throws IOException {
				env.writeln(" ".repeat(count)+"[FIL] "+file.getFileName());
				return FileVisitResult.CONTINUE;
			}
			
		};
		
		try { Files.walkFileTree(f.toPath(), visitor); }
		catch (IOException e) {  }
		
		return ShellStatus.CONTINUE;
	}

	/**
	 * Getter method for the tree command's name;
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
