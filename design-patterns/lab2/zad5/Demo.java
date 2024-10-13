package zad5;

public class Demo {

	public static void main(String[] args) {
		
		Izvor i = new TipkovnickiIzvor();
		SlijedBrojeva sb = new SlijedBrojeva(i);
		
		IListener mean = new MeanListener(sb);
		IListener median = new MedianListener(sb);
		IListener sum = new SumListener(sb);
		sb.addListener(mean);
		sb.addListener(median);
		sb.addListener(sum);
		
		sb.kreni();
	}
	
}
