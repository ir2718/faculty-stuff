package hr.fer.zemris.lsystems.impl;

import java.awt.Color;
import hr.fer.oprpp1.custom.collections.Dictionary;
import hr.fer.oprpp1.math.Vector2D;
import hr.fer.zemris.lsystems.LSystem;
import hr.fer.zemris.lsystems.LSystemBuilder;
import hr.fer.zemris.lsystems.Painter;
import hr.fer.zemris.lsystems.impl.commands.*;

import static java.lang.Math.*;

/**
 * The class represents an builder
 * for an L-system.
 * 
 * @author Ivan Rep
 */
public class LSystemBuilderImpl implements LSystemBuilder {

	private Vector2D origin = new Vector2D(0, 0);
	private double angle = 0;
	private double unitLength = 0.1;
	private double unitLengthDegreeScaler = 1;
	private String axiom = "";
	private Dictionary<Character, String> productions = new Dictionary<>();
	private Dictionary<Character, Command> registeredCommands = new Dictionary<>();

	/**
	 * The method creates a new L-system.
	 * 
	 * @return a new L-system
	 */
	@Override
	public LSystem build() {
		return new LSystemImpl();
	}

	/**
	 * The method parses the given text and assigns the
	 * parameters for an L-system from the text.
	 * 
	 * @param arg0 the string that is parsed
	 * @return a new LSystemBuilder type object
	 * @throws IllegalArgumentException if the input can't be parsed
	 */
	@Override
	public LSystemBuilder configureFromText(String[] arg0) {

		int i=0;
		int len=arg0.length;

		while(i<len) {
			String[] arrOfData =  replaceSpacesAndSplit( arg0[i++] );

			if( arrOfData[0].equals("") ) continue;
			try {

				switch( arrOfData[0] ) {
				case "origin" -> setOrigin( Double.parseDouble( arrOfData[1] ),  Double.parseDouble( arrOfData[2] ) );
				case "angle" -> setAngle( Double.parseDouble(arrOfData[1]) );
				case "unitLength" -> setUnitLength( Double.parseDouble(arrOfData[1]) );
				case "unitLengthDegreeScaler" -> setUnitLengthDegreeScaler( findUnitLenDegScaler(arrOfData) );
				case "command" -> findLengthAndSetCommand(arrOfData);
				case "axiom" -> setAxiom(arrOfData[1]);
				case "production" -> registerProduction( arrOfData[1].charAt(0), arrOfData[2] );
				default -> throw new IllegalArgumentException("The action \""+arrOfData[0]+"\"is not supported.");
				}

			} catch (RuntimeException exc) {
				throw new IllegalArgumentException("At least one of the rows is incorrectly written.");
			}

		}

		return this;
	}

	/**
	 * Finds the length of the given array and sets the command.
	 * Used in configuring from text.
	 * 
	 * @param arrOfData the given array of read data
	 * @throws IllegalArgumentExceotion if the length of the array is not 3 or 4
	 */
	private void findLengthAndSetCommand(String[] arrOfData) {
		switch(arrOfData.length) {
		case 3 	-> registerCommand( arrOfData[1].charAt(0), arrOfData[2] );
		case 4 	-> registerCommand( arrOfData[1].charAt(0), arrOfData[2]+" "+arrOfData[3] );
		default	-> throw new IllegalArgumentException("The command doesn't contain 3 or 4 words.");
		}

	}

	/**
	 * The method parses the unitLengthDegreeScaler 
	 * and returns a double representation of it.
	 * 
	 * @param unitLenDegScalerArr an array containing the line 
	 * of the unitLengthDegreeScaler
	 * @return a double representation of the unitLengthDegreeScaler
	 */
	private static double findUnitLenDegScaler(String[] unitLenDegScalerArr) {
		StringBuilder sb = new StringBuilder();

		for( int i=1, m=unitLenDegScalerArr.length; i<m; i++ )
			sb.append(unitLenDegScalerArr[i]);

		String s = sb.toString();

		if( s.contains("/") )
			return Double.parseDouble( s.substring(0, s.indexOf("/") ) ) / 
					Double.parseDouble( s.substring(s.indexOf("/")+1) );

		return Double.parseDouble( s );
	}

	/**
	 * The method replaces the blanks in a given string by a single space
	 * and returns an array split by the space.
	 * 
	 * @param str the given string
	 * @return a string array that split blanks
	 */
	private static String[] replaceSpacesAndSplit( String str ) {
		return str.replaceAll("\\s+", " ").split(" ");
	}

	/**
	 * The method puts the given key and value into
	 * the dictionary of registered commands.
	 * 
	 * @param arg0 the key character 
	 * @param arg1 the string representation of the command
	 * @return the current LSystemBuilder
	 */
	@Override
	public LSystemBuilder registerCommand(char arg0, String arg1) {
		String[] s = replaceSpacesAndSplit(arg1);

		switch( s[0] ) {
		case	"color" -> registeredCommands.put( arg0, new ColorCommand( Color.decode("0x"+s[1]) ) );
		case	"draw" 	-> registeredCommands.put( arg0, new DrawCommand( Double.parseDouble(s[1]) ) );
		case	"pop" 	-> registeredCommands.put( arg0, new PopCommand() );
		case	"push" 	-> registeredCommands.put( arg0, new PushCommand() );
		case	"rotate"-> registeredCommands.put( arg0, new RotateCommand( Double.parseDouble(s[1]) ) );
		case	"scale" -> registeredCommands.put( arg0, new ScaleCommand( Double.parseDouble( s[1] ) ) );
		case	"skip" 	-> registeredCommands.put( arg0, new SkipCommand( Double.parseDouble(s[1]) ) );
		default 		-> throw new IllegalArgumentException("The command "+s[0]+" isn't supported.");
		}

		return this;
	}

	/**
	 * The method puts the given key and value into
	 * the dictionary of registered productions.
	 * 
	 * @param arg0 the key character 
	 * @param arg1 the value string
	 * @return the current LSystemBuilder
	 */
	@Override
	public LSystemBuilder registerProduction(char arg0, String arg1) {
		productions.put(arg0, arg1);
		return this;
	}

	/**
	 * The method sets the angle
	 * to the given value.
	 * 
	 * @param arg0 the given angle
	 * @return the current LSystemBuilder
	 */
	@Override
	public LSystemBuilder setAngle(double arg0) {
		angle = arg0;
		return this;
	}

	/**
	 * The method sets the axiom
	 * to the given value.
	 * 
	 * @param arg0 the given axiom
	 * @return the current LSystemBuilder
	 */
	@Override
	public LSystemBuilder setAxiom(String arg0) {
		axiom = arg0;
		return this;
	}

	/**
	 * The method sets the origin
	 * to the given value.
	 * 
	 * @param arg0 the given origin
	 * @return the current LSystemBuilder
	 */
	@Override
	public LSystemBuilder setOrigin(double arg0, double arg1) {
		origin = new Vector2D(arg0, arg1);
		return this;
	}

	/**
	 * The method sets the unitLength
	 * to the given value.
	 * 
	 * @param arg0 the given unitLength
	 * @return the current LSystemBuilder
	 */
	@Override
	public LSystemBuilder setUnitLength(double arg0) {
		unitLength = arg0;
		return this;
	}

	/**
	 * The method sets the unitLengthDegreeScaler
	 * to the given value.
	 * 
	 * @param arg0 the given UnitLengthDegreeScaler
	 * @return the current LSystemBuilder
	 */
	@Override
	public LSystemBuilder setUnitLengthDegreeScaler(double arg0) {
		unitLengthDegreeScaler = arg0;
		return this;
	}


	/**
	 * The class represents an object that
	 * implements an L-system.
	 * 
	 * @author Ivan Rep
	 */
	private class LSystemImpl implements LSystem {

		/**
		 * The method draws the L-system
		 * with the given depth.
		 * 
		 * 
		 * @param arg0 the given depth
		 * @param painter this parameter is not used
		 */
		@Override
		public void draw(int arg0, Painter painter) {
			Context ctx = new Context();
			Vector2D direction = new Vector2D( cos(toRadians(angle)), sin(toRadians(angle)) );
			TurtleState t = new TurtleState( origin.copy(), direction.copy(), Color.black, 
					unitLength*pow(unitLengthDegreeScaler, arg0) );
			ctx.pushState(t);

			char[] helper = generate(arg0).toCharArray();
			for(char ch : helper) {
				Command c = registeredCommands.get(ch);
				if(c!=null) c.execute(ctx, painter);
			}

		}


		/**
		 * The method generates the L-system
		 * string with the given depth.
		 * 
		 * @param arg0 the given depth
		 * @return the string that represents the L-system with the given depth
		 */
		public String generate(int arg0) {
			if(arg0==0) return axiom;

			char[] helper = axiom.toCharArray();
			StringBuilder sb = new StringBuilder();

			int j, k;
			for( int i=0; i < arg0; i++ ) {

				j=k=0;
				for( int m=helper.length; j<m ; j++) 
					if( productions.get( helper[j] ) != null ) {
						sb.append( new String(helper).substring(k, j) )
						.append( productions.get(helper[j]) );
						k=j+1;
					}
				sb.append(new String(helper).substring(k, helper.length) );

				helper = sb.toString().toCharArray();
				if(i!=arg0-1) sb = new StringBuilder();
			}

			return sb.toString();

		}

	}


}
