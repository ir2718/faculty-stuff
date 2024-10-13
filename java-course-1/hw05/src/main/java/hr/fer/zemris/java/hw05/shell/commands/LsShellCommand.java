package hr.fer.zemris.java.hw05.shell.commands;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.nio.file.attribute.BasicFileAttributeView;
import java.nio.file.attribute.BasicFileAttributes;
import java.nio.file.attribute.FileTime;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Collections;
import java.util.Date;
import java.util.List;

import hr.fer.zemris.java.hw05.shell.*;

/**
 * Class for the command that prints the contents of the given directory.
 * 
 * @author Ivan Rep
 */
public class LsShellCommand implements ShellCommand, QuotationSkipper  {

	private static final String commandName = "ls";
	private static final String[] description = 
		{"The command prints the names and attributes of all files and directories in the given directory.",
		"Syntax: 'ls path' "};


	/**
	 * Method prints the contents of the given directory.
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

		if(args.length>1) {
			env.writeln("Invalid amount of arguments for ls command.");
			return ShellStatus.CONTINUE;
		} else if (!f.isDirectory()) {
			env.writeln("The given file is not a directory.");
			return ShellStatus.CONTINUE;
		}

		File[] files = f.listFiles();

		Arrays.stream(files).forEach( file -> {
			env.write(String.format("%c%c%c%c %10d %s %s %n", file.isDirectory() ? 'd' : '-',
					file.canRead() ? 'r' : '-',
					file.canWrite() ? 'w' : '-',
					file.canExecute() ? 'x' : '-',
					file.isFile() ? file.length() : findFileLength(file),
					getDateAndTime(file),
					file.getName()));
		});

		return ShellStatus.CONTINUE;
	}

	/**
	 * Method gets the creation time details of the given file
	 * in the format of a string.
	 * 
	 * @param file the given file
	 * @return the string representation of the creation time details for the given file
	 */
	private String getDateAndTime(File file) {
		SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
		BasicFileAttributeView faView = Files.getFileAttributeView( file.toPath(), BasicFileAttributeView.class, LinkOption.NOFOLLOW_LINKS );
		BasicFileAttributes attributes = null;
		try { attributes = faView.readAttributes(); }
		catch (IOException e) { }
		FileTime fileTime = attributes.creationTime();
		String formattedDateTime = sdf.format(new Date(fileTime.toMillis()));
		return formattedDateTime;
	}

	/**
	 * Method returns the size of the given file.
	 * Works for directories too.
	 * 
	 * @param file the given file
	 * @return the length of the given file 
	 */
	private long findFileLength(File file) {
		long size = 0;
		for (File f : file.listFiles()) {
			if (f!=null && f.isFile()) size += f.length();
			else if(f!=null) size += findFileLength(f);
		}

		return size;
	}

	/**
	 * Getter method for the ls shell command's name;
	 * 
	 * @return the string representation of the command name
	 */
	@Override
	public String getCommandName() {
		return commandName;
	}

	/**
	 * Retrieves the ls shell command's description.
	 * 
	 * @return the list containing the command description
	 */
	@Override
	public List<String> getCommandDescription() {
		return Collections.unmodifiableList( Arrays.asList(description) );
	}

}
