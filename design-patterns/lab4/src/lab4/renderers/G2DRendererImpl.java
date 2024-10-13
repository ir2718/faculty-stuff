package lab4.renderers;

import java.awt.Color;
import java.awt.Graphics2D;

import lab4.Point;

public class G2DRendererImpl implements Renderer {

	private Graphics2D g2d;

	public G2DRendererImpl(Graphics2D graphics2d) {
		this.g2d = graphics2d;
	}

	@Override
	public void drawLine(Point s, Point e) {
		g2d.setColor(Color.blue);
		g2d.drawLine(s.getX(), s.getY(), e.getX(), e.getY());
	}

	@Override
	public void fillPolygon(Point[] points) {

		int x[] = new int[points.length];
		int y[] = new int[points.length];
		for(int i=0; i<points.length; i++) {
			y[i] = points[i].getY();
			x[i] = points[i].getX();
		}

		g2d.setColor(Color.blue);
		g2d.fillPolygon(x, y, x.length);

		g2d.setColor(Color.red);
		g2d.drawPolygon(x, y, x.length);
	}

}