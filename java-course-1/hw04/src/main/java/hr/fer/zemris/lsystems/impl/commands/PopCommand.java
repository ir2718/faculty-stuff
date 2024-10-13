package hr.fer.zemris.lsystems.impl.commands;

import hr.fer.zemris.lsystems.Painter;
import hr.fer.zemris.lsystems.impl.*;

/**
 * The class represents the command that pops the current state from the stack
 * 
 * @author Ivan Rep
 */
public class PopCommand implements Command{

	/**
	 * The method pops the current state from the stack.
	 * 
	 * @param ctx the context that obtains the stack
	 * @param painter this parameter is not used 
	 */
	@Override
	public void execute(Context ctx, Painter painter) {
		ctx.popState();
	}
	
}
