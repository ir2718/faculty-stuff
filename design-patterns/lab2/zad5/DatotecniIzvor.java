package zad5;

import java.io.File;
import java.io.IOException;
import java.util.Scanner;

public class DatotecniIzvor implements Izvor {

	private Scanner sc;
	
	public DatotecniIzvor(String s) throws IOException {
		sc = new Scanner(new File(s));
	}
	
	@Override
	public int getNumber() {
		return sc.hasNextInt() ? sc.nextInt() : -1;
	}

}