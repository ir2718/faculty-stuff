package zad4;

public class Sequence implements IGenerate {
	
	private int a;
	private int b;
	private int step;
	
	public Sequence(int a, int b, int step) {
		this.a = a;
		this.b = b;
		this.step = step;
	}

	@Override
	public int[] generate() {
		
		int size = (b-a)/step+1;
		int[] arr = new int[size];
		for(int i=0; i<size; i++)
			arr[i] = a + step * i;
		
		return arr;
	
	}
	
}
