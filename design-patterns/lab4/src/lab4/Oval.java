package lab4;

import java.util.LinkedList;
import java.util.List;
import java.util.Stack;

import lab4.renderers.Renderer;

import static java.lang.Integer.parseInt;
import static java.lang.Math.*;

public class Oval extends AbstractGraphicalObject {

	private Point center;
	
	public Oval() {
		super(new Point[] {new Point(0, 10), new Point(10, 0)});
		this.center = new Point(this.getHotPoint(0).getX(), this.getHotPoint(1).getY());
	}

	public Oval(Point[] hotPoints) {
		super(hotPoints);
		this.center = new Point(this.getHotPoint(0).getX(), this.getHotPoint(1).getY());
	}

	@Override
	public double getHotPointDistance(int index, Point mousePoint) {
		return GeometryUtil.distanceFromPoint(this.getHotPoint(index), mousePoint);
	}

	@Override
	public Rectangle getBoundingBox() {
		Point s = this.getHotPoint(0);
		Point e = this.getHotPoint(1);
		return new Rectangle(2*s.getX() - e.getX(), 2*e.getY() - s.getY(),
				(e.getX() - s.getX())*2,
				(s.getY() - e.getY())*2);
	}

	@Override
	public double selectionDistance(Point mousePoint) {
		return GeometryUtil.distanceFromPoint(this.center, mousePoint);
	}

	@Override
	public String getShapeName() {
		return "Oval";
	}

	@Override
	public void translate(Point other) {
		super.translate(other);
		this.center = this.center.translate(other);
	}
	
	@Override
	public GraphicalObject duplicate() {
		return new Oval();
	}

	@Override
	public void render(Renderer r) {
		double x0 = this.getHotPoint(0).getX();
		double y0 = this.getHotPoint(0).getY();
		double x1 = this.getHotPoint(1).getX();
		double y1 = this.getHotPoint(1).getY();
		
		this.center.setX((int)min(x0,x1));
		this.center.setY((int)min(y0,y1));
		
		int startX = (int)(x0-(x1-x0));
//		int startY = (int)(y1-(y0-y1));
		int endX = (int)(x1);
//		int endY = (int)(y0);

		double centerX = this.center.getX();
		double centerY = this.center.getY();
		
		double b = y0-y1;
		double a = x1-x0;
		
		List<Point> points = new LinkedList<>();
		List<Point> points2 = new LinkedList<>();
		for(int i=startX; i<endX; i++) {
			int currX = (int) (i - centerX);
			int coord = (int) (b*sqrt(1-(pow(currX, 2)/pow(a,2))));
			int yStart = (int) (centerY - coord);
			int yEnd = (int) (centerY + coord);
			points.add(new Point(i, yStart));
			points2.add(new Point(i, yEnd));
		}
			
		Point[] ptsArr = new Point[points.size() + points2.size()];
		ptsArr = points.toArray(ptsArr);

		for(int i=points.size(), j=points2.size()-1; i<ptsArr.length; i++, j--)
			ptsArr[i] = points2.get(j); 
		
		r.fillPolygon(ptsArr);
	}

	@Override
	public String getShapeID() {
		return "@OVAL";
	}

	@Override
	public void load(Stack<GraphicalObject> stack, String data) {
		String[] s = data.split(" ");
		this.setHotPoint(0, new Point(parseInt(s[2]), parseInt(s[3])));
		this.setHotPoint(1, new Point(parseInt(s[0]), parseInt(s[1])));
		stack.add(this);
	}

	@Override
	public void save(List<String> rows) {
		Point s = this.getHotPoint(1);
		Point e = this.getHotPoint(0);
		rows.add(getShapeID()+" "+s.getX()+" "+s.getY()+" "+e.getX()+" "+e.getY());
	}

}
