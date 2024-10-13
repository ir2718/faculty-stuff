package hr.fer.zemris.java.gui.charts;

import java.awt.Color;
import java.awt.Font;
import static java.lang.Math.*;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.geom.AffineTransform;
import java.util.List;
import javax.swing.JComponent;

/**
 * Draws the actual bar chart component.
 * 
 * @author Ivan Rep
 */
public class BarChartComponent extends JComponent {

	public int X_ORIGIN;
	public int Y_ORIGIN;
	private int offset;

	private static final long serialVersionUID = 1L;
	private BarChart barChart;

	/**
	 * Constructor. Sets the origin values and the offset.
	 * 
	 * @param chart the component is of this bar chart object
	 */
	public BarChartComponent(BarChart chart) {
		this.barChart = chart;
		offset = getFontMetrics(new Font("Courier", Font.BOLD, 15)).stringWidth(String.valueOf(barChart.getyMax()));
		X_ORIGIN = 60 + offset;
		Y_ORIGIN = 400;
	}

	/**
	 * Draws the graph using a Graphics object.
	 * 
	 * @param g	Graphics object used for drawing
	 */
	@Override
	protected void paintComponent(Graphics g) {

		g.setColor(Color.white);
		g.fillRect(0, 0, getWidth(), getHeight());
		
		// lines on y axis and grid
		int currH = getHeight()-50;
		int minH = 60;
		int HeightOfOne = (currH-minH)/( (barChart.getyMax()-barChart.getyMin()) / barChart.getDiffY() );
		int cacheYMin=-1;
		int cacheYMax=-1;
		
		for(int i=barChart.getyMin(); i<=barChart.getyMax(); i+=barChart.getDiffY()) {
			if(i==barChart.getyMin()) cacheYMin = currH;
			else if(i==barChart.getyMax()) cacheYMax = currH;

			g.setColor(new Color(157, 157, 157));
			g.drawLine(X_ORIGIN-7, currH, X_ORIGIN, currH);

			g.setColor(Color.black);
			g.setFont(new Font("Courier", Font.BOLD, 12));

			int num = (int) ( 20 + floor( log10( i==0 ? 1 : i  ) )*7.5 );
			g.drawString( String.valueOf( i ), (int) (X_ORIGIN - num ), currH+5);

			g.setColor(new Color(244, 119, 72));
			if(i!=0) g.drawLine(X_ORIGIN, currH, getWidth()-50+offset , currH);

			currH -= HeightOfOne;
		}

		// axes x and y
		g.setColor(new Color(157, 157, 157));
//		g.fillRect(X_ORIGIN, getHeight()-50-1, getWidth()-50-X_ORIGIN , 3);
//		g.fillRect(X_ORIGIN-1, getHeight()-(getHeight()-50), 3 , getHeight()-50-50);
		g.drawLine(X_ORIGIN, getHeight()-50, getWidth()-40+offset , getHeight()-50);
		g.drawLine(X_ORIGIN, getHeight() - 50, X_ORIGIN , 50);

		// horizontal triangle
		int[] xPoints = {getWidth()-40+offset, getWidth()-40+offset, getWidth()-30+offset};
		int[] yPoints = {getHeight()-55, getHeight()-45, getHeight()-50};
		g.fillPolygon( xPoints, yPoints, 3);

		// vertical triangle
		int[] xPoints2 = {X_ORIGIN-5, X_ORIGIN+5, X_ORIGIN};
		int[] yPoints2 = {50, 50, 40};
		g.fillPolygon( xPoints2, yPoints2, 3);


		// lines on x axis and finishing grid
		int currW = getWidth()-50;
		int minW = 60;
		int WidthOfOne = (currW-minW)/barChart.getList().size();
		int cacheXMin=-1;
		int cacheXMax=-1;
		List<XYValue> list = barChart.getList();
		for(int i=0; i<=list.size(); i++) {
			if(i==0) cacheXMin = X_ORIGIN+i*WidthOfOne;
			else if(i==list.size()) cacheXMax = X_ORIGIN+i*WidthOfOne;

			g.setColor(new Color(157, 157, 157));
//			g.fillRect(X_ORIGIN+i*WidthOfOne-1, getHeight()-50, 3, 7);
			g.drawLine(X_ORIGIN+i*WidthOfOne, getHeight()-50, X_ORIGIN+i*WidthOfOne, getHeight()-43);

			g.setColor(new Color(244, 119, 72));
			if(i!=0) g.drawLine(X_ORIGIN+i*WidthOfOne, getHeight() - 50, X_ORIGIN+i*WidthOfOne , 50);
		}

		// draw histogram
		g.setColor(Color.black);
		List<XYValue> values = barChart.getList();
		double notScaledWidth = 1.0/values.size();
		for(int i=0; i<values.size(); i++ ) {
			XYValue val = values.get(i);

			double notScaledHeight = (double) val.getY() / ( barChart.getyMax()-barChart.getyMin() );
			int scaledHeight = (int) (notScaledHeight * (getHeight()-50-60));

			int scaledWidth = (int)((getWidth()-50-60)*notScaledWidth);

			g.setColor(new Color(197, 195, 195));
			g.setXORMode(new Color(191, 184, 174));
			g.fillRect(X_ORIGIN+scaledWidth*i+4, getHeight()-50-scaledHeight+5, scaledWidth-1, scaledHeight-5);
			g.setPaintMode();
			g.setColor(Color.black);
			String text = String.valueOf(val.getX());
			g.setFont(new Font("Courier", Font.BOLD, 12));
			g.drawString( text, X_ORIGIN+scaledWidth*i+scaledWidth/2, getHeight()-35);

			g.setColor(new Color(244, 119, 72));
			g.fillRect(X_ORIGIN+scaledWidth*i, getHeight()-50-scaledHeight, scaledWidth-1, scaledHeight);

			g.setColor(Color.white);
			g.drawLine(X_ORIGIN+scaledWidth*(i+1)-1, getHeight()-50-scaledHeight+5, X_ORIGIN+scaledWidth*(i+1)-1, getHeight()-50);
			
		}
		
		
		// draw line chart
//		Graphics2D g2 = (Graphics2D) g; 
//		g2.setColor(Color.black);
//		List<XYValue> values = barChart.getList();
//		g2.setStroke( new BasicStroke(3) );
//		double notScaledWidth = 1.0/values.size();
//		for(int i=0; i<values.size()-1; i++ ) {
//			XYValue valFirst = values.get(i);
//			XYValue valSecond = values.get(i+1);
//
//			double notScaledHeight = (double) valFirst.getY() / ( barChart.getyMax()-barChart.getyMin() );
//			int scaledHeight = (int) (notScaledHeight * (getHeight()-50-60));
//			int scaledWidth = (int)((getWidth()-50-60)*notScaledWidth);
//
//			double notScaledHeight2 = (double) valSecond.getY() / ( barChart.getyMax()-barChart.getyMin() );
//			int scaledHeight2 = (int) (notScaledHeight2 * (getHeight()-50-60));
//			int scaledWidth2 = (int)((getWidth()-50-60)*notScaledWidth);
//
//			String text = String.valueOf(valFirst.getX());
//			g2.setFont(new Font("Courier", Font.BOLD, 12));
//			g2.drawString( text, X_ORIGIN+scaledWidth*i+scaledWidth/2, getHeight()-35);
//			if(i==values.size()-2) {
//				text = String.valueOf(valSecond.getX());
//				g2.setFont(new Font("Courier", Font.BOLD, 12));
//				g2.drawString( text, X_ORIGIN+scaledWidth*(i+1)+scaledWidth/2, getHeight()-35);
//			}
//
//			g2.drawLine(X_ORIGIN+scaledWidth*i+scaledWidth/2, getHeight()-50-scaledHeight+5, 
//					X_ORIGIN+scaledWidth2*(i+1)+scaledWidth2/2, getHeight()-50-scaledHeight2+5);
//
//		}	


		AffineTransform at = new AffineTransform();
		at.rotate( -PI / 2);
		Graphics2D g2 = (Graphics2D) g;
		g2.setColor(Color.black);
		
		g2.setFont(new Font("Courier", Font.PLAIN, 12));
		int translateX = (int) (getFontMetrics(getFont()).stringWidth(barChart.getDescX()));
		g2.drawString( barChart.getDescX(), (int) ( (cacheXMin+cacheXMax)/2.0 - translateX/2.0 ), getHeight()-15 );
		
		
		g2.setTransform(at);
		int translateY = (int) (getFontMetrics(getFont()).stringWidth(barChart.getDescY()));	
		g2.drawString( barChart.getDescY(), (int) (-cacheYMax-(cacheYMax+cacheYMin)/2.0-translateY/2.0) , 25);

	}

}
