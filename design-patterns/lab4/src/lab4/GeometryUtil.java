package lab4;

import static java.lang.Math.*;


public class GeometryUtil {
	
	public static double distanceFromPoint(Point point1, Point point2) {
		return sqrt(pow((point1.getX() - point2.getX()), 2)
				+ pow((point1.getY() - point2.getY()), 2));
	}
	
	public static double distanceFromLineSegment(Point s, Point e, Point p) {
		// ax + by + c = 0
		boolean sxGreater = s.getX() < e.getX();
		Point start = sxGreater ? s : e;
		Point end = sxGreater ? e : s;

		double a = -(end.getY() - start.getY())/(end.getX() - start.getX());
		double b = 1.0;
		double c = -a*start.getX() - start.getY();
		double x0 = (double) p.getX();
		double y0 = (double) p.getY();
		
		double xNew = (b*(b*x0 - a*y0) - a*c)/(pow(a, 2) + pow(b, 2));
		double yNew = (a*(-b*x0+a*y0))/(pow(a, 2) + pow(b, 2));
		
		if(start.getX() > xNew || start.getY() > yNew)
			return distanceFromPoint(p, start);
		else if(end.getX() < xNew || end.getY() < yNew)
			return distanceFromPoint(p, end);

		return abs(a*x0 + b*y0 + c)/sqrt(pow(a, 2) + pow(b, 2));
	}
}