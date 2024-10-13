package lab4;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import lab4.listeners.DocumentModelListener;
import lab4.listeners.GraphicalObjectListener;

public class DocumentModel {

	public final static double SELECTION_PROXIMITY = 10;
	
	// Kolekcija svih grafičkih objekata:
	private List<GraphicalObject> objects = new ArrayList<>();
	
	// Read-Only proxy oko kolekcije grafičkih objekata:
	private List<GraphicalObject> roObjects = Collections.unmodifiableList(objects);
	
	// Kolekcija prijavljenih promatrača:
	private List<DocumentModelListener> listeners = new ArrayList<>();
	
	// Kolekcija selektiranih objekata:
	private List<GraphicalObject> selectedObjects = new ArrayList<>();
	
	// Read-Only proxy oko kolekcije selektiranih objekata:
	private List<GraphicalObject> roSelectedObjects = Collections.unmodifiableList(selectedObjects);
	
	private final GraphicalObjectListener goListener = new GraphicalObjectListener() {

		@Override
		public void graphicalObjectChanged(GraphicalObject go) {
			notifyListeners();
		}

		@Override
		public void graphicalObjectSelectionChanged(GraphicalObject go) {
			int index = selectedObjects.indexOf(go);
			if(go.isSelected() && index == -1) 
				selectedObjects.add(go);
			else if (!go.isSelected() && index != -1) 
				selectedObjects.remove(go);
			notifyListeners();
		}
	};
	
	public DocumentModel() {}

	public void clear() {
		this.objects.forEach(o -> o.removeGraphicalObjectListener(goListener));
		this.objects.clear();
		this.selectedObjects.clear();
		this.listeners.clear();
		this.notifyListeners();
	}

	public void addGraphicalObject(GraphicalObject obj) {
		objects.add(obj);
		obj.addGraphicalObjectListener(goListener);
		notifyListeners();
	}

	public void removeGraphicalObject(GraphicalObject obj) {
		selectedObjects.remove(obj);
		obj.removeGraphicalObjectListener(goListener);
		objects.remove(obj);
		notifyListeners();
	}
	
	public void deselect() {
		while(selectedObjects.size() > 0) {
			selectedObjects.get(0).setSelected(false);
		}
	}

	public List<GraphicalObject> list() {
		return Collections.unmodifiableList(objects);
	}

	public void addDocumentModelListener(DocumentModelListener l) { 
		listeners.add(l);
	}

	public void removeDocumentModelListener(DocumentModelListener l) {
		listeners.remove(l);
	}

	public void notifyListeners() { 
		this.listeners.forEach(l -> l.documentChange());
	}

	public List<GraphicalObject> getSelectedObjects() { 
		return roSelectedObjects;
	}

	public void increaseZ(GraphicalObject go) {
		int currIndex = objects.indexOf(go);
		objects.remove(go);
		objects.add(currIndex == objects.size() ? currIndex : currIndex + 1, go);
		notifyListeners();
	}

	public void decreaseZ(GraphicalObject go) {
		int currIndex = objects.indexOf(go);
		objects.remove(go);
		objects.add(currIndex == 0 ? currIndex : currIndex - 1, go);
		notifyListeners();
	}

	public GraphicalObject findSelectedGraphicalObject(Point mousePoint) {
		GraphicalObject found = null;
		for(GraphicalObject o : objects) {
			double d = o.selectionDistance(mousePoint);
			if(d < SELECTION_PROXIMITY) {
				found = o;
				break;
			}
		}
		return found; 
	}

	public int findSelectedHotPoint(GraphicalObject object, Point mousePoint) {
		int value = -1;
		double currValue = Double.MAX_VALUE;
		for(GraphicalObject o : objects) {
			for(int i = 0; i<object.getNumberOfHotPoints(); i++) {
				double d = o.getHotPointDistance(i, mousePoint);
				if(d < SELECTION_PROXIMITY && d <= currValue) {
					value = i;
					currValue = d;
				}
			}
		}
		return value; 
	}

}
