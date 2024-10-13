package MyTextEditor;

import java.util.LinkedList;
import java.util.List;
import java.util.Stack;

public class UndoManager {
	
	public static final UndoManager UNDO_MANAGER = new UndoManager();

	private Stack<EditAction> undoStack;
	private Stack<EditAction> redoStack;
	
	private List<SelectionObserver> undoObservers;
	private List<SelectionObserver> redoObservers;
	
	private UndoManager() {
		this.undoStack = new Stack<>();
		this.redoStack = new Stack<>();
		this.undoObservers = new LinkedList<>();
		this.redoObservers = new LinkedList<>();
	}
	
	public static UndoManager getInstance() {
		return UNDO_MANAGER;
	}
	
	public Stack<EditAction> getUndoStack() {
		return undoStack;
	}

	public Stack<EditAction> getRedoStack() {
		return redoStack;
	}
	
	public void undo() {
		if(this.undoStack.isEmpty()) return;
			
		EditAction action = this.undoStack.pop();
		this.redoStack.push(action);
		action.executeUndo();
	}
	
	public void redo() {
		if(this.redoStack.isEmpty()) return;
		
		EditAction action = this.redoStack.pop();
		this.undoStack.push(action);
		action.executeDo();
	}
	
	public void push(EditAction c) {
		this.redoStack.clear();
		this.undoStack.push(c);
	}
	
	public void addUndoObserver(SelectionObserver o) {
		this.undoObservers.add(o);
	}
	
	public void removeUndoObserver(SelectionObserver o) {
		this.undoObservers.remove(o);
	}
	
	public void addRedoObserver(SelectionObserver o) {
		this.redoObservers.add(o);
	}
	
	public void removeRedoObserver(SelectionObserver o) {
		this.redoObservers.remove(o);
	}
	
	public void notifyUndoObservers() {
		this.undoObservers.forEach(o -> o.update());
	}
	
	public void notifyRedoObservers() {
		this.redoObservers.forEach(o -> o.update());
	}
}
