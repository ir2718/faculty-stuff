package lab4.states;

import java.awt.event.KeyEvent;
import java.util.LinkedList;
import java.util.List;

import lab4.DocumentModel;
import lab4.GraphicalObject;
import lab4.Point;
import lab4.Rectangle;
import lab4.composite.CompositeShape;
import lab4.renderers.Renderer;

public class SelectShapeState implements State {

	private DocumentModel model;

	public SelectShapeState(DocumentModel model) {
		this.model = model;
	}

	@Override
	public void mouseDown(Point mousePoint, boolean shiftDown, boolean ctrlDown) {
		GraphicalObject go = model.findSelectedGraphicalObject(mousePoint);
		if(go == null) return;

		if(!ctrlDown) model.deselect();

		go.setSelected(true);
	}

	@Override
	public void mouseUp(Point mousePoint, boolean shiftDown, boolean ctrlDown) { }

	@Override
	public void mouseDragged(Point mousePoint) {
		if(model.getSelectedObjects().size() == 1) {
			GraphicalObject go = this.model.getSelectedObjects().get(0);
			int index = model.findSelectedHotPoint(go, mousePoint);
			if(index == -1) return;
			Point hpSelected = go.getHotPoint(index);
			hpSelected.setCoordinates(mousePoint);
			model.notifyListeners();
		}
	}

	@Override
	public void keyPressed(int keyCode) {
		List<GraphicalObject> gos = model.getSelectedObjects();
		if(gos.size() == 1) {
			switch(keyCode) {
			case KeyEvent.VK_ADD -> model.increaseZ(gos.get(0));
			case KeyEvent.VK_SUBTRACT -> model.decreaseZ(gos.get(0));
			}
			model.notifyListeners();
		}

		switch(keyCode) {
		case KeyEvent.VK_UP -> gos.forEach(o -> o.translate(new Point(0, -1)));
		case KeyEvent.VK_DOWN -> gos.forEach(o -> o.translate(new Point(0, 1)));
		case KeyEvent.VK_LEFT ->gos.forEach(o -> o.translate(new Point(-1, 0)));
		case KeyEvent.VK_RIGHT -> gos.forEach(o -> o.translate(new Point(1, 0)));

		case KeyEvent.VK_G -> {
			if (model.getSelectedObjects().size() > 1) {
				List<GraphicalObject> l = model.getSelectedObjects();
				GraphicalObject go = new CompositeShape(l);
				while(!l.isEmpty())
					model.removeGraphicalObject(l.get(0));
				model.addGraphicalObject(go);
				go.setSelected(true);
			}
		}
		
		case KeyEvent.VK_U -> {
			if (model.getSelectedObjects().size() == 1 
					&& model.getSelectedObjects().get(0).getShapeName().equals("Composite")) {
				CompositeShape cs = (CompositeShape)model.getSelectedObjects().get(0);
				model.removeGraphicalObject(cs);
				for(GraphicalObject o : cs.getChildren()) {
					model.addGraphicalObject(o);
					o.setSelected(true);
				}
			}
		}
		}

		model.notifyListeners();
	}

	@Override
	public void afterDraw(Renderer r, GraphicalObject go) {
		if(model.getSelectedObjects().size() == 1 && go.isSelected()) {
			Rectangle r2 = go.getBoundingBox();
			drawSquare(r, r2);
			for(int i=0; i<go.getNumberOfHotPoints(); i++) {
				Point p = go.getHotPoint(i);
				Rectangle r3 = new Rectangle(p.getX()-4, p.getY()-4, 8, 8);
				drawSquare(r, r3);
			}
		} else {
			for(GraphicalObject go2 : model.getSelectedObjects()) {
				Rectangle r2 = go2.getBoundingBox();
				drawSquare(r, r2);
			}		
		}
	}

	@Override
	public void afterDraw(Renderer r) {}

	@Override
	public void onLeaving() {
		model.deselect();
		model.notifyListeners();
	}

	private void drawSquare(Renderer r, Rectangle r2) {
		r.drawLine(new Point(r2.getX(), r2.getY()), 
				new Point(r2.getX(), r2.getY() + r2.getHeight()));

		r.drawLine(new Point(r2.getX(), r2.getY()), 
				new Point(r2.getX() + r2.getWidth(), r2.getY()));

		r.drawLine(new Point(r2.getX() + r2.getWidth(), r2.getY()), 
				new Point(r2.getX() + r2.getWidth(), r2.getY() + r2.getHeight()));

		r.drawLine(new Point(r2.getX(), r2.getY() + r2.getHeight()), 
				new Point(r2.getX() + r2.getWidth(), r2.getY() + r2.getHeight()));
	}

}
