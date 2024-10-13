package hr.fer.zemris.lsystems.impl;

import java.awt.Color;

import hr.fer.oprpp1.math.Vector2D;

/**
 * The class represents a state used in drawing an
 * L-system using a GUI.
 * 
 * @author Ivan Rep
 */
public class TurtleState {
	
	private Vector2D currentLocation;
	private Vector2D direction;
	private Color drawingColor;
	private double effectiveDistance;
	
	/**
	 * The constructor sets the current location, the direction, the color and the effective distance parameters.
	 * 
	 * @param currentLocation the given current location vector
	 * @param direction the given direction vector 
	 * @param drawingColor the given drawing color
	 * @param effectiveDistance the given effective distance
	 */
	public TurtleState( Vector2D currentLocation, Vector2D direction, Color drawingColor, double effectiveDistance ) {
		this.currentLocation = currentLocation; // maybe needed .copy()
		this.direction = direction;				// maybe needed .copy()
		this.drawingColor = drawingColor;
		this.effectiveDistance = effectiveDistance;
	}
	
	/**
	 * The method returns the copy of this state.
	 * 
	 * @return a new TurtleState object with the same parameters as the current one
	 */
	public TurtleState copy() {
		return new TurtleState( currentLocation.copy(), direction.copy(), drawingColor, effectiveDistance );
	}
	
	/**
	 * The method gets the current location of the current state.
	 * 
	 * @return the current location of the current state
	 */
	public Vector2D getCurrentLocation() {
		return this.currentLocation;
	}

	/**
	 * The method gets the direction of the current state.
	 * 
	 * @return the direction of the current state
	 */
	public Vector2D getDirection() {
		return this.direction;
	}
	
	/**
	 * The method gets the color of the current state.
	 * 
	 * @return the color of the current state
	 */
	public Color getDrawingColor() {
		return this.drawingColor;
	}
	
	/**
	 * The method sets the drawing color parameter to the given color.
	 * 
	 * @param drawingColor the given color
	 */
	public void setDrawingColor(Color drawingColor) {
		this.drawingColor = drawingColor;
	}
	
	/**
	 * The method gets the effective distance of the current state.
	 * 
	 * @return the effective distance of the current state
	 */
	public double getEffectiveDistance() {
		return this.effectiveDistance;
	}
	
	/**
	 * The method sets the effective distance to the given effective distance.
	 * 
	 * @param effectiveDistance the given effective distance
	 */
	public void setEffectiveDistance(double effectiveDistance) {
		this.effectiveDistance = effectiveDistance;
	}
}
