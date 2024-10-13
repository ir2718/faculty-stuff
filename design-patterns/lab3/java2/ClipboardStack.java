package MyTextEditor;

import java.util.LinkedList;
import java.util.List;
import java.util.Stack;

public class ClipboardStack {

	private Stack<String> texts;
	
	private List<ClipboardObserver> observers;
	
	public ClipboardStack() {
		this.texts = new Stack<>();
		this.observers = new LinkedList<>();
	}
	
	public void push(String s) {
		this.texts.push(s);
	}
	
	public String pop() {
		return this.texts.pop();
	}
	
	public String peek() {
		return this.texts.peek();
	}
	
	public boolean isEmpty() {
		return this.texts.isEmpty();
	}
	
	public void clear() {
		this.texts.clear();
	}
	
	public void addObserver(ClipboardObserver o) {
		this.observers.add(o);
	}
	
	public void removeObserver(ClipboardObserver o) {
		this.observers.add(o);
	}
	
	public void notifyObservers() {
		this.observers.forEach(o -> o.updateClipboard());
	}
}
