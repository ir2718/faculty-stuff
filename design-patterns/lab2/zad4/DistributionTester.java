package zad4;

import static java.util.Arrays.sort;

public class DistributionTester {

	private IGenerate generator;
	private IFind finder;

	public DistributionTester(IGenerate generator, IFind finder) {
		this.generator = generator;
		this.finder = finder;
	}

	@Override() 
	public String toString() {

		StringBuilder sb = new StringBuilder();
		
		int[] arr = generator.generate();
		sort(arr);
		
		String delimiter = ", ";
		for(int p=10; p<=90; p+=10) 
			sb.append(finder.find(arr, p) + delimiter);
		
		sb.setLength(sb.length() - delimiter.length());
		
		return sb.toString();
	}
}
