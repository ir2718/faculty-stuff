package zad5;

public class SumListener implements IListener {
	
	private SlijedBrojeva subject;
	
	public SumListener(SlijedBrojeva subject) {
		this.subject = subject;
	}

	@Override
	public void update() {
		System.out.println("Sum: "+subject
				.getCollection()
				.stream()
				.mapToInt(i -> Integer.valueOf(i)).sum());
	}

}
