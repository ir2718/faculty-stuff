package lab4.composite;

import static java.lang.Integer.parseInt;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.Stack;
import java.util.stream.Collectors;

import lab4.GraphicalObject;
import lab4.Point;
import lab4.Rectangle;
import lab4.listeners.GraphicalObjectListener;
import lab4.renderers.Renderer;

public class CompositeShape implements GraphicalObject {

	private List<GraphicalObject> children;
	private boolean selected;
	private List<GraphicalObjectListener> listeners;

	public CompositeShape(List<GraphicalObject> list) {
		this.children = new LinkedList<>(list);
		this.selected = false;
		this.listeners = new LinkedList<>();
	}

	public CompositeShape() {
		this.children = new LinkedList<>();
		this.selected = false;
		this.listeners = new LinkedList<>();
	}
	
	public List<GraphicalObject> getChildren() {
		return this.children;
	}

	@Override
	public boolean isSelected() {
		return this.selected;
	}

	@Override
	public void setSelected(boolean selected) {
		this.selected = selected;
		this.notifySelectionListeners();
	}

	@Override
	public int getNumberOfHotPoints() {
		return 0;
	}

	@Override
	public Point getHotPoint(int index) {
		return null;
	}

	@Override
	public void setHotPoint(int index, Point point) { }

	@Override
	public boolean isHotPointSelected(int index) {
		return false;
	}

	@Override
	public void setHotPointSelected(int index, boolean selected) { }

	@Override
	public double getHotPointDistance(int index, Point mousePoint) {
		return -1;
	}

	@Override
	public void translate(Point delta) {
		for(GraphicalObject go : children)
			go.translate(delta);
		notifyListeners();
	}

	@Override
	public Rectangle getBoundingBox() {
		List<Rectangle> rectangles = children.stream().map(o -> o.getBoundingBox()).collect(Collectors.toList());

		List<Integer> x = rectangles.stream().map(o -> o.getX()).collect(Collectors.toList());
		x.addAll(rectangles.stream().map(o -> o.getX() + o.getWidth()).collect(Collectors.toList()));

		List<Integer> y = rectangles.stream().map(o -> o.getY()).collect(Collectors.toList());
		y.addAll(rectangles.stream().map(o -> o.getY() + o.getHeight()).collect(Collectors.toList()));

		int minX = Collections.min(x);
		int maxX = Collections.max(x);
		int minY = Collections.min(y);
		int maxY = Collections.max(y);

		return new Rectangle(minX, minY, maxX-minX, maxY-minY);
	}

	@Override
	public double selectionDistance(Point mousePoint) {
		List<Double> d = new LinkedList<>();
		children.forEach(o -> d.add(o.selectionDistance(mousePoint)));
		return Collections.min(d);
	}

	@Override
	public void render(Renderer r) {
		children.forEach(o -> o.render(r));
	}

	public void notifySelectionListeners() {
		listeners.forEach(l -> l.graphicalObjectSelectionChanged(this));
	}

	public void notifyListeners() {
		listeners.forEach(l -> l.graphicalObjectChanged(this));
	}

	@Override
	public void addGraphicalObjectListener(GraphicalObjectListener l) {
		listeners.add(l);
	}

	@Override
	public void removeGraphicalObjectListener(GraphicalObjectListener l) {
		listeners.remove(l);
	}

	@Override
	public String getShapeName() {
		return "Composite";
	}

	@Override
	public GraphicalObject duplicate() {
		List<GraphicalObject> objects = new LinkedList<>();
		for(GraphicalObject go : children)
			objects.add(go.duplicate());
		return new CompositeShape(objects);
	}

	@Override
	public String getShapeID() {
		return "@COMP";
	}

	
	@Override
	public void load(Stack<GraphicalObject> stack, String data) {
		String s = data.trim();
		List<GraphicalObject> l = new ArrayList<>();
		int i = 0; 
		while (i < parseInt(s)) {
			l.add(stack.pop());
			i++;
		}
		this.children = l;
		stack.push(this);
	}

	@Override
	public void save(List<String> rows) {
		children.forEach(o -> o.save(rows));
		rows.add(getShapeID()+" "+ children.size());
	}

}
