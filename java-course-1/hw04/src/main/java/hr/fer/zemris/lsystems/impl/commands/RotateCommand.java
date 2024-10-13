package hr.fer.zemris.lsystems.impl.commands;

import hr.fer.zemris.lsystems.Painter;
import hr.fer.zemris.lsystems.impl.*;
import static java.lang.Math.toRadians;

/**
 * The class represents the command that rotates the direction vector by an angle.
 * 
 * @author Ivan Rep
 */
public class RotateCommand implements Command {

	private double angle;
	
	/**
	 * The constructor method sets the angle parameter to the given angle value.
	 * 
	 * @param angle the given angle value
	 */
	public RotateCommand(double angle) {
		this.angle = angle;
	}

	/**
	 * The method rotates the current state's direction vector by the angle parameter.
	 * 
	 * @param ctx the context which contains the stack
	 * @param painter this parameter is not used
	 */
	@Override
	public void execute(Context ctx, Painter painter) {
		ctx.getCurrentState().getDirection().rotate( toRadians(angle) );
	}
	
}
