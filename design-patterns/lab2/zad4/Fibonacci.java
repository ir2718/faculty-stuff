package zad4;

public class Fibonacci implements IGenerate {

	private int n;
	
	public Fibonacci(int n) {
		this.n = n;
	}
	
	@Override
	public int[] generate() {
 
		int first = 0, second = 1;
		int copy;
		int[] arr = new int[n];
		
		for(int i=0; i<n; i++) 
			if(i==0) {
				arr[i] = first;
			} else if(i==1) {
				arr[i] = second;
			} else {
				copy = first;
				first = second;
				second += copy;
				arr[i] = second;
			}
		
		
		return arr;
	}
}
