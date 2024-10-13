package hr.fer.oprpp1.math;

import static java.lang.Math.*;

/**
 * The class represents a two dimensional vector that has
 * support for addition, scaling and rotating.
 * 
 * @author Ivan Rep
 */
public class Vector2D {

	private double x;
	private double y;
	
	/**
	 * The constructor method. Sets the parameters to the given values
	 * 
	 * @param x the x coordinate of the vector
	 * @param y the y coordinate of the vector
	 */
	public Vector2D(double x, double y) {
		this.x = x;
		this.y = y;
	}

	/**
	 * The getter method for the x coordinate.
	 * 
	 * @return the x coordinate of the current vector
	 */
	public double getX() {
		return this.x;
	}

	/**
	 * The getter method for the y coordinate.
	 * 
	 * @return the y coordinate of the current vector
	 */
	public double getY() {
		return this.y;
	}
	
	/**
	 * Adds the given vector to the current vector.
	 * 
	 * @param offset the given vector
	 */
	public void add(Vector2D offset) {
		this.x += offset.x;
		this.y += offset.y;
	}
	
	/**
	 * Returns a new vector that has the x and y coordinate equal to the
	 * result of adding the current vector and the given one.
	 * 
	 * @param offset the given vector
	 * @return a new object of type Vector2D with the computed values
	 */
	public Vector2D added(Vector2D offset) {
		return new Vector2D( this.x+offset.x, this.y+offset.y );
	}
	
	/**
	 * Rotates the current vector by an angle in radians.
	 * 
	 * @param angle the rotation angle
	 */
	public void rotate(double angle) {
		double copyX = this.x;
		this.x = x*cos(angle) - y*sin(angle);
		this.y = copyX*sin(angle) + y*cos(angle);
	}
	
	/**
	 * Returns a new vector that is a rotated version
	 * of this vector by the given angle in radians.
	 * 
	 * @param angle the rotation angle
	 * @return a new object of type Vector2D with the computed values
	 */
	public Vector2D rotated(double angle) {
		return new Vector2D(x*cos(angle) - y*sin(angle), x*sin(angle) + y*cos(angle));
	}

	/**
	 * Scales the current vector by a scalar.
	 * 
	 * @param scalar the scaling coefficient
	 */
	public void scale(double scaler) {
		this.x *= scaler;
		this.y *= scaler;
	}
	
	/**
	 * Scales the current vector by a scalar and returns
	 * a new object of type Vector2D.
	 * 
	 * @param scalar the scaling coefficient
	 * @return a new object of type Vector2D with the computed values
	 */
	public Vector2D scaled(double scaler) {
		return new Vector2D(this.x * scaler, this.y * scaler);
	}
	

	/**
	 * Returns a new object of type Vector2D that 
	 * has the same values as the current one.
	 * 
	 * @return a new Object of type Vector2D that has the same values
	 */
	public Vector2D copy() {
		return new Vector2D(this.x, this.y);
	}


	/**
	 * Returns true if the x and y coordinate of the current vector and
	 * the second vector are the same.
	 * 
	 * @param obj the second vector
	 * @return true if the coordinates are the same otherwise false
	 */
	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		Vector2D other = (Vector2D) obj;
		
		return (x-other.x) < pow(10, -10) 
				&& (y-other.y) < pow(10, -10)  ;
	}
	
	
	
}
