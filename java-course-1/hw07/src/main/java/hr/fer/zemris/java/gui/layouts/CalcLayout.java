package hr.fer.zemris.java.gui.layouts;

import java.awt.Component;
import static java.lang.Math.*;
import java.awt.Container;
import java.awt.Dimension;
import java.awt.Insets;
import java.awt.LayoutManager2;
import java.util.HashMap;
import java.util.Map;
import static hr.fer.zemris.java.gui.layouts.RCPosition.*;

/**
 * Class represents a calculator layout.
 * 
 * @author Ivan Rep
 */
public class CalcLayout implements LayoutManager2 {
	
	public final static int ROWS = 5;
	public final static int COLUMNS = 7;
	public final static int FIRST_SIZE = 5;

	private int offset;
	private Map<RCPosition, Component> map;

	/**
	 * Empty constructor. Sets offset to 0.
	 */
	public CalcLayout() {
		this(0);
	}

	/**
	 * Constructor. Sets offset to given value
	 * 
	 * @param offset the given offset value
	 */
	public CalcLayout(int offset) {
		super();
		this.offset = offset;
		this.map = new HashMap<>();
	}

	/**
	 * Unsupported method. Throws an exception.
	 * 
	 * @param name	
	 * @param comp	
	 * @throws UnsupporteedOperationException whenever it's called
	 */
	@Override
	public void addLayoutComponent(String name, Component comp) { 
		throw new UnsupportedOperationException("This action is not supported.");
	}
	
	/**
	 * Removes the layout component from the current layout.
	 */
	@Override
	public void removeLayoutComponent(Component comp) { 
		map.values().removeIf(comp::equals);
	}

	/**
	 * Adds layout component from the string or the given RCPosition.
	 * 
	 * @param comp			component that represents the given string or RCPosition object
	 * @param constraints	the given string or RCPosition object
	 * @throws IllegalArgumentException	if the given object is neither a String nor an RCPosition
	 * @throws NullPointerException		if the given constraints are null
	 * @throws CalcLayoutException		if the given position is invalid
	 */
	@Override
	public void addLayoutComponent(Component comp, Object constraints) {
		if(constraints == null || comp == null) throw new NullPointerException("No argument can be null.");

		RCPosition rcp = null;

		if( constraints instanceof String )
			rcp = parse( (String) constraints );
		else if( constraints instanceof RCPosition )
			rcp = (RCPosition) constraints;
		else
			throw new IllegalArgumentException("The given object "
					+ "is neither String nor RCPosition.");

		if( rcp.getRow()<1 || rcp.getColumn()<1 ||
				rcp.getRow()>ROWS || rcp.getColumn()>COLUMNS )
			throw new CalcLayoutException("The given position "
					+ "can't have negative coordinates.");

		if ( rcp.getRow()==1 && rcp.getColumn()>=2 && rcp.getColumn()<=5 )
			throw new CalcLayoutException("The given "
					+ "position is invalid.");

		if( map.containsKey(rcp) )
			throw new CalcLayoutException("There's already a component on "
					+ "the coordinates: "+rcp+".");

		map.put(rcp, comp);
	}


	/**
	 * Calculates the maximum size of the layout.
	 * 
	 * @param target maximum size of layout of this container is calculated
	 * @return a new Dimension object with the calculated values
	 */
	@Override
	public Dimension maximumLayoutSize(Container target) {
		
		ILayoutSize ilayout = (c) -> c.getMaximumSize();
		return ilayout.FindLayoutSize(target, this);
		
	}

	/**
	 * Returns 0.
	 */
	@Override
	public float getLayoutAlignmentX(Container target) { return 0; }

	/**
	 * Returns 0.
	 */
	@Override
	public float getLayoutAlignmentY(Container target) { return 0; }

	
	/**
	 * Unsupported method. Throws an exception.
	 * 
	 * @param 	target	
	 * @throws 	UnsupporteedOperationException whenever it's called
	 */
	@Override
	public void invalidateLayout(Container target) { }

	/**
	 * Calculates the preferred size of the layout.
	 * 
	 * @param parent preferred size of layout of this container is calculated
	 * @return a new Dimension object with the calculated values
	 */
	@Override
	public Dimension preferredLayoutSize(Container parent) {
		
		ILayoutSize ilayout = (c) -> c.getPreferredSize();
		return ilayout.FindLayoutSize(parent, this);
		
	}


	/**
	 * Sets the components into the layout. Uses uniform method
	 * 
	 * @param parent	components are added to this container
	 * @see Uniform
	 */
	@Override
	public void layoutContainer(Container parent) {
		System.out.println("Pozvan layoutContainer!");
		Insets ins = parent.getInsets();
		int w = parent.getWidth()-ins.left-ins.right;
		int h = parent.getHeight()-ins.top-ins.bottom;
		RCPosition oneOne = RCPosition.parse("1, 1");

		double heightOfOne = ( h-((double)ROWS-1)*offset )/ROWS; 
		double widthOfOne = (w-((double)COLUMNS-1)*offset )/COLUMNS;
		
		int[] heights = Uniform(h-offset*(ROWS-1), ROWS);
		int[] widths = Uniform(w-offset*(COLUMNS-1), COLUMNS);

		for( Map.Entry<RCPosition, Component> entry : map.entrySet()  ) {
			RCPosition rcp = entry.getKey();
			Component c = entry.getValue();
			
			if( rcp.equals( oneOne ) )
				c.setBounds( ins.left,
						ins.top,
						(int) (widthOfOne*FIRST_SIZE+(FIRST_SIZE-1)*offset),
						(int) heightOfOne
						);
			else
				c.setBounds( (int) ( (widthOfOne+offset)*(rcp.getColumn()-1) + ins.left),
						(int) ( (heightOfOne+offset)*(rcp.getRow()-1) + ins.top),
						widths[rcp.getColumn()-1],
						heights[rcp.getRow()-1]
						);
		}

	}
	
	/**
	 * Returns an array of calculated uniformly distributed integer values for width or height.
	 * 
	 * @param pixels	amount of pixels that are available
	 * @param constant	constant for how many values the array contains
	 * @return			an array of calculated uniformly distributed values
	 */
	private static int[] Uniform(int pixels, int constant) {
		
		int sizeOfLesser = (int) floor( (double)pixels/constant );
		int amountOfGreater = pixels % constant;
		int amountOfLesser = constant - amountOfGreater;
		boolean moreGreater = amountOfGreater > amountOfLesser;
		int bound = moreGreater ? amountOfLesser : amountOfGreater;
		int[] retArr = new int[constant];
		
		for(int i = 1, count=0; count < bound; i+=2, count++) 
			retArr[i] = moreGreater ? sizeOfLesser : sizeOfLesser + 1;
		
		for(int i = 0; i<constant; i++)
			if( retArr[i]==0 ) retArr[i] = moreGreater ? sizeOfLesser + 1 : sizeOfLesser;
			
		return retArr;	
	}


	/**
	 * Calculates the minimum size of the layout.
	 * 
	 * @param parent minimum size of layout of this container is calculated
	 * @return a new Dimension object with the calculated values
	 */
	@Override
	public Dimension minimumLayoutSize(Container parent) {
		
		ILayoutSize ilayout = (c) -> c.getMaximumSize();
		return ilayout.FindLayoutSize(parent, this);
	
	}
	
	/**
	 * Interface used as strategy pattern for defining the layout size.
	 * 
	 * @author Ivan Rep
	 */
	 private interface ILayoutSize {

		 	/**
		 	 * Finds the layout size for the given container and layout.
		 	 * 
		 	 * @param parent	the given container
		 	 * @param l			the given layout
		 	 * @return			new Dimension object with the computed values
		 	 */
			default Dimension FindLayoutSize(Container parent, CalcLayout l) {
				Component[] comps = parent.getComponents();
				Component first = l.map.get(RCPosition.parse("1, 1"));
				int maxH = 0;
				int maxW = 0;

				for( Component c : comps ) {
					
					if ( getComponent(c)!=null && ( ( first!=null  && !first.equals(c) ) || first==null ) && getComponent(c).height>maxH ) 
						maxH = getComponent(c).height;
					
					if ( getComponent(c)!=null && ( first==null || ( first!=null && !first.equals(c) ) ) && getComponent(c).width>maxW ) 
						maxW = getComponent(c).width;
					else if ( getComponent(c)!=null && first!=null && first.equals(c) && CalculateSizeFromFirst(c, this, l)>maxW )
						maxW = CalculateSizeFromFirst(c, this, l);
					
				
				}

				Insets ins = parent.getInsets();
				
				return new Dimension( maxW*COLUMNS+(COLUMNS-1)*l.offset + ins.left + ins.right,
						maxH*ROWS+(ROWS-1)*l.offset + ins.bottom + ins.top );
			}

			
			/**
			 * Method should return the wanted size of the given component.
			 * 
			 * @param c	the given component
			 * @return	wanted size of the given component
			 */
			abstract Dimension getComponent(Component c);

		}
	 
	 /**
	  * Method used for calculating the size from a component at the position (1,1).
	  * 
	  * @param c			the given container
	  * @param iLayoutSize	wanted size of the given component
	  * @param 				the given layout
	  * @return				calculated size from the first element
	  */
	 private static int CalculateSizeFromFirst(Component c, ILayoutSize iLayoutSize, CalcLayout l) {
		 
		 return (int) ( ( (double)iLayoutSize.getComponent(c).width-(FIRST_SIZE-1)*l.offset ) / FIRST_SIZE );
		 
	 }

}
