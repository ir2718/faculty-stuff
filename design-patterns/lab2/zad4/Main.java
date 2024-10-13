package zad4;

public class Main {

	public static void main(String[] args) {
		
		DistributionTester test1 = new DistributionTester(new Fibonacci(10), new FindInterpolated());
		System.out.println(test1);
		
		DistributionTester test2 = new DistributionTester(new Fibonacci(10), new FindSorted());
		System.out.println(test2);
		
		DistributionTester test3 = new DistributionTester(new Sequence(5, 15, 2), new FindInterpolated());
		System.out.println(test3);
		
		DistributionTester test4 = new DistributionTester(new Sequence(5, 15, 2), new FindSorted());
		System.out.println(test4);
		
		DistributionTester test5 = new DistributionTester(new Normal(0, 15, 5), new FindInterpolated());
		System.out.println(test5);
		
		DistributionTester test6 = new DistributionTester(new Normal(0, 15, 5), new FindSorted());
		System.out.println(test6);
	}
	
}
