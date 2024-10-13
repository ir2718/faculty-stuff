package zad5;

import java.util.Scanner;

public class TipkovnickiIzvor implements Izvor {

	private Scanner sc;

	public TipkovnickiIzvor() {
		this.sc = new Scanner(System.in);
	}

	@Override
	public int getNumber() {
		return sc.hasNextInt() ? sc.nextInt() : -1;
	}

}