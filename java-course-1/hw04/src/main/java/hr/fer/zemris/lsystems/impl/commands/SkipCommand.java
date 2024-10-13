package hr.fer.zemris.lsystems.impl.commands;

import hr.fer.zemris.lsystems.Painter;
import hr.fer.zemris.lsystems.impl.*;

/**
 * The class represents a skip command that moves the
 * current state to a position.
 * 
 * @author Ivan Rep
 */
public class SkipCommand implements Command {
	
	private double step;
	
	/**
	 * The constructor method that sets the step parameter to the given value.
	 * 
	 * @param step the given step value
	 */
	public SkipCommand(double step) {
		this.step = step;
	}

	/**
	 * The method moves the current state's current location to the 
	 * position of the product of effective distance and the step value.
	 * 
	 * @param ctx the context that obtains the stack
	 * @param painter this parameter is not used
	 */
	@Override
	public void execute(Context ctx, Painter painter) {
		
		ctx.getCurrentState().getCurrentLocation().add( 
				ctx.getCurrentState().getDirection()
				.scaled( step*ctx.getCurrentState().getEffectiveDistance() ) );
		
	}

	
}
