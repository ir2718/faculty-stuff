package hr.fer.zemris.lsystems.impl.commands;

import hr.fer.zemris.lsystems.Painter;
import hr.fer.zemris.lsystems.impl.*;

/**
 * The class represents the command that pushes the copy of the current state to the top of the stack.
 * 
 * @author Ivan Rep
 */
public class PushCommand implements Command{

	/**
	 * The method pushes the copy of the current state to the top of the stack.
	 * 
	 * @param ctx the context which contains the stack
	 * @param painter this parameter is not used
	 */
	@Override
	public void execute(Context ctx, Painter painter) {
		ctx.pushState(ctx.getCurrentState().copy());
	}
	
}
