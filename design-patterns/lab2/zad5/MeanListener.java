package zad5;

public class MeanListener implements IListener {

	private SlijedBrojeva subject;
	
	public MeanListener(SlijedBrojeva subject) {
		this.subject = subject;
	}

	@Override
	public void update() {
		System.out.println("Mean: "+subject
				.getCollection()
				.stream()
				.mapToInt(i -> Integer.valueOf(i))
				.average()
				.getAsDouble());
	}

}
