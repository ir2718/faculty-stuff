package lab4.states;

import lab4.DocumentModel;
import lab4.GraphicalObject;
import lab4.Point;
import lab4.renderers.Renderer;

public class AddShapeState implements State {

	private GraphicalObject prototype;
	private DocumentModel model;
	
	public AddShapeState(GraphicalObject go, DocumentModel model) {
		this.prototype = go;
		this.model = model;
	}
	
	@Override
	public void mouseDown(Point mousePoint, boolean shiftDown, boolean ctrlDown) {
		GraphicalObject o = prototype.duplicate();
		o.translate(mousePoint);
		model.addGraphicalObject(o);
	}

	@Override
	public void mouseUp(Point mousePoint, boolean shiftDown, boolean ctrlDown) {
	}

	@Override
	public void mouseDragged(Point mousePoint) {
		
	}

	@Override
	public void keyPressed(int keyCode) {
	}

	@Override
	public void afterDraw(Renderer r, GraphicalObject go) {
	}

	@Override
	public void afterDraw(Renderer r) {
	}

	@Override
	public void onLeaving() {
	} 

}
