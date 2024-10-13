package zad5;

import java.util.List;

public class MedianListener implements IListener {
	
	private SlijedBrojeva subject;

	public MedianListener(SlijedBrojeva subject) {
		this.subject = subject;
	}

	@Override
	public void update() {
		List<Integer> l = subject.getCollection();
		System.out.print("Median: ");
		int size = l.size();
		
		if(size%2 != 0) {
			System.out.println(l.get(size/2));
			return;
		}
		
		System.out.println((l.get(size/2) + l.get(size/2-1))/2.0);
			
	}

}
