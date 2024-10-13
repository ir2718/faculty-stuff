package hr.fer.zemris.java.gui.prim;

import static java.lang.Math.*;

import java.util.ArrayList;
import java.util.List;

import javax.swing.ListModel;
import javax.swing.event.ListDataEvent;
import javax.swing.event.ListDataListener;

/**
 * Class represents a list of prime numbers.
 * 
 * @author Ivan Rep
 */
public class PrimListModel<integer> implements ListModel<Integer> {
	
	private List<Integer> elems = new ArrayList<>();
	private List<ListDataListener> observers = new ArrayList<>();

	private int currPrime;

	/**
	 * Constructor. Sets the current prime to 1.
	 */
	public PrimListModel() {
		currPrime = 1;
	}
	
	/**
	 * Getter method for the current prime.
	 * 
	 * @return returns the current prime
	 */
	public int getCurrPrime() {
		return currPrime;
	}

	/**
	 * Getter method for the size of the list of prime numbers.
	 * 
	 * @return size of the list of prime numbers
	 */
	@Override
	public int getSize() {
		return elems.size();
	}

	/**
	 * Getter method for the integer at the given index.
	 *
	 * @param index the given index	
	 * @return 		the elements at the given index in the list of prime numbers
	 */
	@Override
	public Integer getElementAt(int index) {
		return elems.get(index);
	}

	/**
	 * Adds the given ListDataListener object into the list of observers.
	 * 
	 * @param l the given ListDataListener object
	 */
	@Override
	public void addListDataListener(ListDataListener l) {
		observers.add(l);
	}

	/**
	 * Removes the given ListDataListener object from the list of observers.
	 * 
	 * @param l the given ListDataListener object
	 */
	@Override
	public void removeListDataListener(ListDataListener l) {
		observers.remove(l);
	}
	

	/**
	 * Adds the given Integer object into the list of prime numbers.
	 * 
	 * @param element the given Integer object
	 */
	public void add(Integer element) {
		int pos = elems.size();
		elems.add(element);
		
		ListDataEvent event = new ListDataEvent(this, ListDataEvent.INTERVAL_ADDED, pos, pos);
		for(ListDataListener l : observers) 
			l.intervalAdded(event);

	}
	

	/**
	 * Removes the number at the given position from the list of prime numbers.
	 * 
	 * @param pos the given position
	 */
	public void remove(int pos) {
		elems.remove(pos);
		ListDataEvent event = new ListDataEvent(this, ListDataEvent.INTERVAL_REMOVED, pos, pos);
		for(ListDataListener l : observers) 
			l.intervalRemoved(event);
		
	}
	
	/**
	 * Returns the first prime number after the current prime.
	 * 
	 * @return the next prime number
	 */
	public int next() {

		if(currPrime==1) {
			currPrime = 2;
			return currPrime;
		}

		int k = currPrime;
		int counter;

		while(true) {
			k++;
			counter = 0;
			
			for(int i=2; i<=sqrt(k); i++) 
				if(k%i==0) {
					counter++;
					break;
				}

			if(counter==0) break;
		}

		currPrime = k;
		return k;
	}

}
