package hr.fer.zemris.java.gui.layouts;

import org.junit.jupiter.api.*;
import hr.fer.zemris.java.gui.layouts.*;

import static org.junit.jupiter.api.Assertions.assertThrows;

import javax.swing.JLabel;

public class CalcLayoutExceptionsTest {

	@Test
	public void TestRowLessThan1() {
		CalcLayout l = new CalcLayout();
		assertThrows(CalcLayoutException.class, () -> l.addLayoutComponent(new JLabel("label"), "0, 2"));
	}

	@Test
	public void TestRowGreaterThan5() {
		CalcLayout l = new CalcLayout();
		assertThrows(CalcLayoutException.class, () -> l.addLayoutComponent(new JLabel("label"), "6, 2"));
	}

	@Test
	public void TestColumnLessThan1() {
		CalcLayout l = new CalcLayout();
		assertThrows(CalcLayoutException.class, () -> l.addLayoutComponent(new JLabel("label"), "1, 0"));
	}

	@Test
	public void TestColumnGreaterThan7() {
		CalcLayout l = new CalcLayout();
		assertThrows(CalcLayoutException.class, () -> l.addLayoutComponent(new JLabel("label"), "1, 8"));
	}

	@Test
	public void TestColumnBetween1And6WhenRow1() {
		CalcLayout l = new CalcLayout();
		assertThrows(CalcLayoutException.class, () -> l.addLayoutComponent(new JLabel("label"), "1, 2"));
		assertThrows(CalcLayoutException.class, () -> l.addLayoutComponent(new JLabel("label"), "1, 3"));
		assertThrows(CalcLayoutException.class, () -> l.addLayoutComponent(new JLabel("label"), "1, 4"));
		assertThrows(CalcLayoutException.class, () -> l.addLayoutComponent(new JLabel("label"), "1, 5"));
	}

	@Test
	public void TestRow1Column1WhenItAlreadyExists() {
		CalcLayout l = new CalcLayout();
		l.addLayoutComponent(new JLabel("label"), "1, 1");
		assertThrows(CalcLayoutException.class, () -> l.addLayoutComponent(new JLabel("label2"), "1, 1"));

	}

}
