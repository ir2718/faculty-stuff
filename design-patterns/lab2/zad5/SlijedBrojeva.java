package zad5;

import java.util.LinkedList;
import java.util.List;

public class SlijedBrojeva {

	private List<Integer> collection;
	private List<IListener> listeners;
	private Izvor source;

	public SlijedBrojeva(Izvor source) {
		this.collection = new LinkedList<>();
		this.listeners = new LinkedList<>();
		this.source = source;
	}

	public List<Integer> getCollection() {
		return this.collection;
	}

	public void addListener(IListener l) {
		listeners.add(l);
	}

	public void removeListener(IListener l) {
		listeners.remove(l);
	}

	public void notifyListeners() {
		listeners.stream().forEach(l -> l.update());
	}

	public void kreni() {

		while(true) {
			int num = source.getNumber();
			if(num == -1) return;

			collection.add(num);
			this.notifyListeners();

			try { Thread.sleep(1000); }
			catch (InterruptedException e) { }
		}

	}

}
