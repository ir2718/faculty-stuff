package lab4;

import java.util.ArrayList;
import java.util.List;

import lab4.listeners.GraphicalObjectListener;

public abstract class AbstractGraphicalObject implements GraphicalObject {
	
	private Point[] hotPoints;
	private boolean[] hotPointsSelected;
	private boolean selected;
	private List<GraphicalObjectListener> listeners = new ArrayList<>();
	
	public AbstractGraphicalObject() {}
	
	public AbstractGraphicalObject(Point[] points) {
		this.hotPoints = points;
		this.hotPointsSelected = new boolean[points.length];
		this.selected = false;
	}

	public Point getHotPoint(int i) {
		return hotPoints[i];
	}

	public void setHotPoint(int i, Point hotPoint) {
		this.hotPoints[i] = hotPoint;
		this.notifyListeners();
	}

	public int getNumberOfHotPoints() {
		return this.hotPoints.length;
	}
	
	public boolean isHotPointSelected(int i) {
		return hotPointsSelected[i];
	}
	
	public void setHotPointSelected(int i, boolean b) {
		this.hotPointsSelected[i] = b;
	}
	public boolean isSelected() {
		return selected;
	}

	public void setSelected(boolean selected) {
		this.selected = selected;
		this.notifySelectionListeners();
	}
	
	public void translate(Point other) {
		for(int i=0; i<hotPoints.length; i++)
			hotPoints[i] = hotPoints[i].translate(other);
	}
	
	public void addGraphicalObjectListener(GraphicalObjectListener l) {
		this.listeners.add(l);
	}
	
	public void removeGraphicalObjectListener(GraphicalObjectListener l) {
		this.listeners.remove(l);
	}
	
	public void notifyListeners() {
		this.listeners.forEach(l -> l.graphicalObjectChanged(this));
	}
	
	public void notifySelectionListeners() {
		this.listeners.forEach(l -> l.graphicalObjectSelectionChanged(this));
	}

}
