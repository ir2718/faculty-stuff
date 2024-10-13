package zad4;

import static java.util.Arrays.sort;

public class FindInterpolated implements IFind {

	public FindInterpolated() {}
	
	@Override
	public int find(int[] arr, int p) {
		sort(arr);
		int n = arr.length;
		if(p < 100*0.5/n) 
			return arr[0];
		
		if(p > 100*(n-0.5)/n)
			return arr[n-1];
		
		double pVi, pVi_1;
		for(int i=0; i<n-1; i++) {
			pVi = 100 * (i+0.5)/n;
			pVi_1 = 100 * (i+1.5)/n;
			
			if(p >= pVi && p <= pVi_1)
				return (int) (arr[i] + n * (p - pVi) * (arr[i+1] - arr[i])/100.0);
		}
		
		return 0;
	}
	
}
