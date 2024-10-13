package hr.fer.zemris.lsystems.impl.commands;

import hr.fer.oprpp1.math.Vector2D;
import hr.fer.zemris.lsystems.Painter;
import hr.fer.zemris.lsystems.impl.*;

/**
 * The class represents the command that draws a line 
 * on the GUI.
 * 
 * @author Ivan Rep
 */
public class DrawCommand implements Command {
	
	private double step;
	
	/**
	 * The method assigns the given step value to the step parameter.
	 * 
	 * @param step the given step value
	 */
	public DrawCommand(double step) {
		this.step = step;
	}

	/**
	 * The method draws a line using the current location vector
	 * and the current location vector translated by 
	 * the direction vector scaled by the step.
	 * 
	 * @param ctx the context that obtains the stack
	 * @param painter the parameter used to draw on the GUI
	 */
	@Override
	public void execute(Context ctx, Painter painter) {
		
		Vector2D begin = ctx.getCurrentState().getCurrentLocation().copy();
		
		ctx.getCurrentState().getCurrentLocation().add( 
				ctx.getCurrentState().getDirection().copy()
				.scaled(step*ctx.getCurrentState().getEffectiveDistance()) );
		
		Vector2D end = ctx.getCurrentState().getCurrentLocation();
		
		painter.drawLine( begin.getX() ,
				begin.getY(),
				end.getX(),
				end.getY(),
				ctx.getCurrentState().getDrawingColor(), 1f);
		
	}

	
}
