package MyTextEditor;

import java.util.Collection;
import java.util.Iterator;

public class MyIterator<T> implements Iterator<T> {
	
	private int index;
	private T[] col;
	
	@SuppressWarnings("unchecked")
	public MyIterator(Collection<T> col) {
		this.col = (T[]) col.toArray();
		this.index = 0;
	}

	@Override
	public boolean hasNext() {
		return this.index < this.col.length;
	}

	@Override
	public T next() {
		return this.col[index++];
	}
	
	public T first() {
		return this.col[0];
	}
	
	public T last() {
		return this.col[col.length - 1];
	}
}
