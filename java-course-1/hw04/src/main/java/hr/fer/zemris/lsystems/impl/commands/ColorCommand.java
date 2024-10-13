package hr.fer.zemris.lsystems.impl.commands;

import java.awt.Color;

import hr.fer.zemris.lsystems.Painter;
import hr.fer.zemris.lsystems.impl.*;

/**
 * The class represents a command that sets the drawing color.
 * 
 * @author Ivan Rep
 */
public class ColorCommand implements Command{

	private Color color;
	
	/**
	 * The constructor assigns the given value to
	 * the parameter color
	 * 
	 * @param color the color parameter
	 */
	public ColorCommand(Color color) {
		this.color = color;
	}

	/**
	 * The method sets the color of the state
	 * on the top of the stack.
	 * 
	 * @param ctx the context which contains the stack
	 * @param painter this parameter is not used
	 */
	@Override
	public void execute(Context ctx, Painter painter) {
		ctx.getCurrentState().setDrawingColor(color);
	}
	
	
}
