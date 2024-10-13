package hr.fer.zemris.java.hw05.shell;

import java.util.Collections;
import java.util.NoSuchElementException;
import java.util.Scanner;
import java.util.SortedMap;
import java.util.TreeMap;
import hr.fer.zemris.java.hw05.shell.commands.*;

/**
 * An implementation of the environment interface.
 * 
 * @author Ivan Rep
 */
public class ShellEnvironment implements Environment {

	private SortedMap<String, ShellCommand> commands;
	private Scanner scan;

	private Character promptSymbol = '>';
	private Character multilineSymbol = '|';
	private Character morelinesSymbol = '\\';

	/**
	 * Constructor method for the environment. Sets
	 * the values of the variables and commands.
	 */
	public ShellEnvironment() {
		this.scan = new Scanner(System.in);
		TreeMap<String, ShellCommand> commands = new TreeMap<>();
		commands.put("exit", new ExitShellCommand());
		commands.put("ls", new LsShellCommand());
		commands.put("charsets", new CharsetsCommand());
		commands.put("cat", new CatCommand());
		commands.put("tree", new TreeCommand());
		commands.put("copy", new CopyCommand());
		commands.put("mkdir", new MkDirCommand());
		commands.put("hexdump", new HexDumpCommand());
		commands.put("help", new HelpCommand());
		commands.put("symbol", new SymbolCommand());
		this.commands = Collections.unmodifiableSortedMap(commands);
	}

	/**
	 * Reads a line of input from the console.
	 * 
	 * @return returns a string representation of a line from the input
	 */
	@Override
	public String readLine() throws ShellIOException {
		String ln = null;
		while(ln==null) {
			ln = scan.nextLine();
		}
		return ln;
	}

	/**
	 * Method writes the given text to the console.
	 * 
	 * @param text the given text
	 * @throws ShellIOException if there is an error while writing
	 */
	@Override
	public void write(String text) throws ShellIOException {
		System.out.print(text);
	}

	/**
	 * Method writes the given text to the console and adds
	 * a newline at the end.
	 * 
	 * @param text the given text
	 * @throws ShellIOException if there is an error while writing
	 */
	@Override
	public void writeln(String text) throws ShellIOException {
		System.out.println(text);
	}

	/**
	 * Method returns an immutable map of the available commands.
	 * 
	 * @return immutable map of available commands
	 */
	@Override
	public SortedMap<String, ShellCommand> commands() {
		return this.commands;
	}

	/**
	 * Getter method for the multiline symbol.
	 * 
	 * @return the Character representation of the multiline symbol
	 */
	@Override
	public Character getMultilineSymbol() {
		return this.multilineSymbol;
	}

	/**
	 * Setter method for the multiline symbol.
	 * 
	 * @param symbol the new value for the multiline symbol
	 */
	@Override
	public void setMultilineSymbol(Character symbol) {
		this.multilineSymbol=symbol;
	}

	/**
	 * Getter method for the prompt symbol.
	 * 
	 * @return the Character representation of the prompt symbol
	 */
	@Override
	public Character getPromptSymbol() {
		return this.promptSymbol;
	}

	/**
	 * Setter method for the prompt symbol.
	 * 
	 * @param symbol the new value for the prompt symbol
	 */
	@Override
	public void setPromptSymbol(Character symbol) {
		this.promptSymbol=symbol;
	}

	/**
	 * Getter method for the morelines symbol.
	 * 
	 * @return the Character representation of the morelines symbol
	 */
	@Override
	public Character getMorelinesSymbol() {
		return this.morelinesSymbol;
	}

	/**
	 * Setter method for the morelines symbol.
	 * 
	 * @param symbol the new value for the morelines symbol
	 */
	@Override
	public void setMorelinesSymbol(Character symbol) {
		this.morelinesSymbol=symbol;
	}

	/**
	 * Method closes the scanner.
	 */
	public void exit() {
		scan.close();
	}

}
