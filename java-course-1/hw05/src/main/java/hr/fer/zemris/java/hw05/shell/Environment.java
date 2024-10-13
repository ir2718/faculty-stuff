package hr.fer.zemris.java.hw05.shell;

import java.util.SortedMap;


public interface Environment {

	/**
	 * Method should read the line from the input.
	 * 
	 * @return the string representation of the line read from the input
	 * @throws ShellIOException if there is an error while reading
	 */
	String readLine() throws ShellIOException;
	
	/**
	 * Method should write the given text to the user.
	 * 
	 * @param text the given text
	 * @throws ShellIOException if there is an error while writing
	 */
	void write(String text) throws ShellIOException;
	
	/**
	 * Method should write the given text to the user and adds
	 * a newline at the end.
	 * 
	 * @param text the given text
	 * @throws ShellIOException if there is an error while writing
	 */
	void writeln(String text) throws ShellIOException;
	
	/**
	 * Method should return an immutable map of the available commands.
	 * 
	 * @return immutable map of available commands
	 */
	SortedMap<String, ShellCommand> commands();
	
	/**
	 * Getter method for the multiline symbol.
	 * 
	 * @return the Character representation of the multiline symbol
	 */
	Character getMultilineSymbol();
	
	/**
	 * Setter method for the multiline symbol.
	 * 
	 * @param symbol the new value for the multiline symbol
	 */
	void setMultilineSymbol(Character symbol);
	
	/**
	 * Getter method for the prompt symbol.
	 * 
	 * @return the Character representation of the prompt symbol
	 */
	Character getPromptSymbol();
	
	/**
	 * Setter method for the prompt symbol.
	 * 
	 * @param symbol the new value for the prompt symbol
	 */
	void setPromptSymbol(Character symbol);
	
	/**
	 * Getter method for the morelines symbol.
	 * 
	 * @return the Character representation of the morelines symbol
	 */
	Character getMorelinesSymbol();
	
	/**
	 * Setter method for the morelines symbol.
	 * 
	 * @param symbol the new value for the morelines symbol
	 */
	void setMorelinesSymbol(Character symbol);
	

}

