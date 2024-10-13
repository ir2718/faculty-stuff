package lab4.renderers;

import lab4.Point;

public interface Renderer {
	
	void drawLine(Point s, Point e);
	void fillPolygon(Point[] points);

}
