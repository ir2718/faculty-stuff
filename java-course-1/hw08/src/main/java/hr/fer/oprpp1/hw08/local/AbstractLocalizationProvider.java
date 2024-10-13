package hr.fer.oprpp1.hw08.local;

import java.util.ArrayList;
import java.util.List;

public abstract class AbstractLocalizationProvider implements ILocalizationProvider {

	private List<ILocalizationListener> listeners= new ArrayList<>();
	
	public void addLocalizationListener(ILocalizationListener l) {
		listeners.add(l);
	}
	
	public void removeLocalizationListener(ILocalizationListener l) {
		listeners.remove(l);
	}
	
	public void fire() {
		listeners.iterator().forEachRemaining( l -> l.localizationChanged() );
	}
	
	
}
