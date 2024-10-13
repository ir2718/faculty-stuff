package zad4;

public class FindSorted implements IFind {
	
	public FindSorted() {}
	
	@Override
	public int find(int[] arr, int p) {
		return arr[(int) (p*arr.length/100.0 + 0.5) - 1];
	}
}
