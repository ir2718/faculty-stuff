package podzadatak1;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.FontMetrics;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Insets;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.awt.event.WindowEvent;
import java.awt.event.WindowListener;

import javax.swing.JButton;
import javax.swing.JComponent;
import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.SwingUtilities;
import javax.swing.WindowConstants;

public class Frame extends JFrame {
	
	public static void main(String[] args) {
		SwingUtilities.invokeLater(new Runnable() {
			@Override
			public void run() {
				new Frame().setVisible(true);
			}
		});
	}
	
	public Frame() {
		setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);
		setLocation(0, 0);
		setSize(1000, 800);
		TestComponent tc = new TestComponent();
		this.add(tc);
		KeyListener kl = new KeyListener() {

			public void keyTyped(KeyEvent e) { }

			public void keyReleased(KeyEvent e) { }
			
			@Override
			public void keyPressed(KeyEvent e) {
				if(e.getKeyCode() == KeyEvent.VK_ENTER)
					tc.getKeyListener().keyPressed(e);
			}
		};
		this.addKeyListener(kl);
	}

	
	@Override
	public void paint(Graphics g) {
		super.paint(g);
	}

}

class TestComponent extends JComponent {
	
	private KeyListener kl;
	
	public TestComponent() { }
	
	public KeyListener getKeyListener()  {
		return kl;
	}
	
	@Override
	public void paintComponent(Graphics g) {
		Graphics2D g2 = (Graphics2D) g;
		g2.setColor(Color.red);
		g2.setStroke(new BasicStroke(1));
		g2.drawLine(50, 200, 500, 200);
		g2.drawLine(200, 50, 200, 500);
		
		g2.setColor(Color.black);
		Insets ins = getInsets();
		FontMetrics fm = g2.getFontMetrics();
		g2.drawString("Some text", ins.left, fm.getHeight()*1+ins.top);
		g2.drawString("This is some other text", ins.left, fm.getHeight()*2+ins.top);
		
		kl = new KeyListener() {
			@Override
			public void keyTyped(KeyEvent e) { }
			
			@Override
			public void keyReleased(KeyEvent e) { }
			
			@Override
			public void keyPressed(KeyEvent e) {
				if(e.getKeyCode()==KeyEvent.VK_ENTER)
					System.exit(0);
			}
		};
		this.addKeyListener(kl);
	}
	
}
