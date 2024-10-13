package zad4;

import java.util.Random;

public class Normal implements IGenerate {
	
	private double mu;
	private double sigma;
	private int n;
	
	public Normal(double mu, double sigma, int n) {
		this.mu = mu;
		this.sigma = sigma;
		this.n = n;
	}

	@Override
	public int[] generate() {
		Random r = new Random();
		
		int[] arr = new int[n];
		for(int i=0; i<n; i++) 
			arr[i] = (int) (mu + sigma * r.nextGaussian());
		
		return arr;
	}


}
