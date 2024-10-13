package hr.fer.zemris.ooup.lab2.model;

import java.io.File;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.net.MalformedURLException;
import java.net.URL;
import java.net.URLClassLoader;

public class AnimalFactory {
	
	public static void main(String[] args) throws Exception {
		Animal a =newInstance("Parrot", "bobi");
		a.animalPrintGreeting();
		a.animalPrintMenu();
	}
	
	public static Animal newInstance(String animalKind, String name) throws Exception {
		Class<Animal> clazz = null;
		
//		ClassLoader parent = AnimalFactory.class.getClassLoader();
//		URLClassLoader newClassLoader = new URLClassLoader(
//			new URL[] {
//				new File("D:/java/plugins/").toURI().toURL(),
//				new File("D:/java/plugins-jarovi/zivotinje.jar").toURI().toURL()
//			}, parent);
//		clazz = (Class<Animal>)newClassLoader.loadClass("hr.fer.zemris.ooup.lab2.model.plugins."+animalKind);
		
		
		clazz = (Class<Animal>)Class.forName("hr.fer.zemris.ooup.lab2.model.plugins."+animalKind);
		Constructor<?> ctr = clazz.getConstructor(String.class);
		Animal animal = (Animal)ctr.newInstance(name);
		
		return animal;
		
	}

}
