package hr.fer.zemris.ooup.lab2.model.plugins;

import hr.fer.zemris.ooup.lab2.model.Animal;

public class Parrot extends Animal {
	
	private String name;
	
	public Parrot() {}
	
	public Parrot(String name) {
		this.name = name;
	}
	
	@Override
	public String name() {
		return this.name;
	}

	@Override
	public String greet() {
		return "squawk";
	}

	@Override
	public String menu() {
		return "sjemenke";
	}

}
