package hr.fer.zemris.lsystems.impl;

import hr.fer.zemris.lsystems.Painter;

/**
 * The interface models a command
 * used in drawing an L-system.
 * 
 * @author Ivan Rep
 */
public interface Command {

	/**
	 * The method should execute
	 * do a certain action depending
	 * on the type of command.
	 * 
	 * @param ctx the context of the current stack
	 * @param painter the parameter used for drawing lines
	 */
	void execute(Context ctx, Painter painter);
	
}
