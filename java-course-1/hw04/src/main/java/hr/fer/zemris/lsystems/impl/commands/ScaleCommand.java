package hr.fer.zemris.lsystems.impl.commands;

import hr.fer.zemris.lsystems.Painter;
import hr.fer.zemris.lsystems.impl.*;

/**
 * The class represents the command that scales the current states effective distance.
 * 
 * @author Ivan Rep
 */
public class ScaleCommand implements Command{

	private double factor;
	
	/**
	 * The constructor method that assigns the given factor value to the factor parameter.
	 * 
	 * @param factor the given factor value
	 */
	public ScaleCommand(double factor) {
		this.factor = factor;
	}
	
	/**
	 * The method scales the current state's effective distance parameter by the factor value.
	 * 
	 * @param ctx the context which contains the stack
	 * @param painter this parameter is not used here
	 */
	@Override
	public void execute(Context ctx, Painter painter) {
		TurtleState temp = ctx.getCurrentState();
		temp.setEffectiveDistance(temp.getEffectiveDistance()*factor);
	}

	
	
}
