package zad6;

import java.util.LinkedList;
import java.util.List;

public class Cell implements ISubject, IListener {
	
	private String coordinate;
	private String exp;
	private int value;
	
	private List<IListener> listeners;
	private Sheet sheet;
	
	public Cell(Sheet sheet) {
		this.exp = "0";
		this.value = 0;
		this.listeners = new LinkedList<>();
		this.sheet = sheet;
	}
	
	public void setCoordinate(String coordinate) {
		this.coordinate = coordinate;
	}
	
	public String getCoordinate() {
		return this.coordinate;
	}

	@Override
	public String toString() {
		return Integer.toString(this.value);
	}
	
	@Override
	public void update() {
		this.setValue(this.sheet.evaluate(this));
	}

	public void setExp(String exp) {
		this.exp = exp;
	}
	
	public void setValue(int value) {
		this.value = value;
		this.notifyListeners();
	}
	
	public String getExp() {
		return this.exp;
	}
	
	public int getValue() {
		return this.value;
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + ((coordinate == null) ? 0 : coordinate.hashCode());
		return result;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		Cell other = (Cell) obj;
		if (coordinate == null) {
			if (other.coordinate != null)
				return false;
		} else if (!coordinate.equals(other.coordinate))
			return false;
		return true;
	}

	@Override
	public void addListener(IListener l) {
		listeners.add(l);
	}

	@Override
	public void removeListener(IListener l) {
		listeners.remove(l);
	}

	@Override
	public void notifyListeners() {
		listeners.stream().forEach(l -> l.update());
	}
	
}
