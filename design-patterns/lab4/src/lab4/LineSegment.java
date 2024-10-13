package lab4;

import static java.lang.Integer.parseInt;

import java.util.List;
import java.util.Stack;

import lab4.renderers.Renderer;

public class LineSegment extends AbstractGraphicalObject {

	public LineSegment() {
		super(new Point[] { new Point(0,0),  new Point(10,0)});
	}

	public LineSegment(Point start, Point end) {
		super(new Point[] {start, end});
	}

	@Override
	public double getHotPointDistance(int index, Point mousePoint) {
		return GeometryUtil.distanceFromPoint(this.getHotPoint(index), mousePoint);
	}

	@Override
	public Rectangle getBoundingBox() {
		Point s = this.getHotPoint(0);
		Point e = this.getHotPoint(1);

		int ey = e.getY();
		int ex = e.getX();
		int sy = s.getY();
		int sx = s.getX();

		if(sx < ex && sy < ey)
			return new Rectangle(sx, sy, ex-sx, ey-sy);

		if(sx > ex && sy < ey) 
			return new Rectangle(ex, sy, sx-ex, ey-sy);


		if(sx < ex && sy > ey)
			return new Rectangle(sx, ey, ex-sx, sy-ey);

		return new Rectangle(ex, sy, sx-ex, ey-sy);
	}

	@Override
	public double selectionDistance(Point mousePoint) {
		return GeometryUtil.distanceFromLineSegment(this.getHotPoint(0), 
				this.getHotPoint(1), mousePoint);
	}

	@Override
	public String getShapeName() {
		return "Linija";
	}

	@Override
	public GraphicalObject duplicate() {
		return new LineSegment();
	}

	@Override
	public void render(Renderer r) {
		r.drawLine(this.getHotPoint(0), this.getHotPoint(1));
	}

	@Override
	public String getShapeID() {
		return "@LINE";
	}

	@Override
	public void load(Stack<GraphicalObject> stack, String data) {
		String[] s = data.split(" ");
		stack.add(new LineSegment(new Point(parseInt(s[0]), parseInt(s[1])), 
				new Point(parseInt(s[2]), parseInt(s[3]))));
	}

	@Override
	public void save(List<String> rows) {
		Point s = this.getHotPoint(0);
		Point e = this.getHotPoint(1);
		rows.add(getShapeID()+" "+s.getX()+" "+s.getY()+" "+e.getX()+" "+e.getY());
	}

}
