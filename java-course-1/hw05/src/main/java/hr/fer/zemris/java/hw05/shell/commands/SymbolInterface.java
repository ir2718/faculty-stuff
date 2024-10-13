package hr.fer.zemris.java.hw05.shell.commands;

import hr.fer.zemris.java.hw05.shell.Environment;
import hr.fer.zemris.java.hw05.shell.ShellStatus;

/**
 * The interface is a contract for symbols.
 * Handles the changing of a symbol and printing it to the console.
 * 
 * @author Ivan Rep
 */
public interface SymbolInterface {

	/**
	 * Method checks which action will happen.
	 * Used for setting or getting a symbol.
	 * 
	 * @param args the string containing everything after the key word
	 * @param env the environment of the shell
	 */
	default void findSymbolCommand(String[] args, Environment env) {
		
		if( args.length==2 && args[1].length()==1 ) {
			env.writeln("Symbol for "+args[0]+" changed from '"+getMethod()+"' to '"+args[1]+"'.");
			setMethod( args[1].charAt(0) );
		} else if (args.length==1 )
			env.writeln("Symbol for "+args[0]+" is '"+getMethod()+"'.");
		else 
			env.writeln("The symbol "+getMethod()+" command was incorrectly written.");
		
	}

	/**
	 * Method should get the current value of the symbol
	 * 
	 * @return the character representation of the symbol
	 */
	abstract Character getMethod();
	
	/**
	 * Method should set the current value of the symbol
	 * to the given value.
	 * 
	 * @param c the given value
	 */
	abstract void setMethod( char c );

}
