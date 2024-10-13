package lab4.states;

import java.util.LinkedList;
import java.util.List;

import lab4.DocumentModel;
import lab4.GraphicalObject;
import lab4.Point;
import lab4.renderers.Renderer;

public class EraserState implements State {
	
	private DocumentModel model;
	private List<Point> points;
	
	public EraserState(DocumentModel model) {
		this.model = model;
		this.points = new LinkedList<>();
	}

	@Override
	public void mouseDown(Point mousePoint, boolean shiftDown, boolean ctrlDown) { }

	@Override
	public void mouseUp(Point mousePoint, boolean shiftDown, boolean ctrlDown) {
		for(Point p : points) {
			GraphicalObject go = model.findSelectedGraphicalObject(p);
			if (go != null) model.removeGraphicalObject(go);
		}
		points.clear();
	}

	@Override
	public void mouseDragged(Point mousePoint) {
		points.add(mousePoint);
		model.notifyListeners();
	}

	@Override
	public void keyPressed(int keyCode) { }

	@Override
	public void afterDraw(Renderer r, GraphicalObject go) {
		for(int i=0; i<points.size()-1; i++) {
			r.drawLine(points.get(i), points.get(i+1));
		}
	}

	@Override
	public void afterDraw(Renderer r) { }

	@Override
	public void onLeaving() { }

}
