package lab4;

import java.awt.BorderLayout;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionAdapter;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Stack;

import javax.swing.JButton;
import javax.swing.JComponent;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JOptionPane;
import javax.swing.JToolBar;

import lab4.composite.CompositeShape;
import lab4.renderers.G2DRendererImpl;
import lab4.renderers.Renderer;
import lab4.renderers.SVGRendererImpl;
import lab4.states.AddShapeState;
import lab4.states.EraserState;
import lab4.states.IdleState;
import lab4.states.SelectShapeState;
import lab4.states.State;

public class Frame extends JFrame {
 
	public static void main(String[] args) {
		Frame f = new Frame();
		f.setSize(500, 500);
		f.setVisible(true);
		
		List<GraphicalObject> objects = new ArrayList<>();
		objects.add(new LineSegment(new Point(200, 200), new Point(200, 250)));
		objects.add(new Oval(new Point[] {new Point (300, 350), new Point(350, 300)}));
		GUI gui = new GUI(objects);
		
		f.add(gui);
		f.setVisible(true);
		f.addKeyListener(gui.kl);
		f.setDefaultCloseOperation(EXIT_ON_CLOSE);
	}
	
	
	private static class GUI extends JComponent {

		private List<GraphicalObject> objects;
		private DocumentModel model;
		private State currentState;
		
		public GUI(List<GraphicalObject> objects) {
			this.objects = objects;
			this.model = new DocumentModel();
			this.model.addDocumentModelListener(() -> repaint());
			this.setLayout(new BorderLayout());
			this.setVisible(true);
			this.setFocusable(true);

			JToolBar toolbar = new JToolBar();
			
			for(GraphicalObject o : this.objects) {
				JButton b = new JButton(o.getShapeName());
				b.addActionListener(new ActionListener() {
					@Override
					public void actionPerformed(ActionEvent e) {
						currentState.onLeaving();
						currentState = new AddShapeState(o, model);
					}
				});
				b.setFocusable(false);
			
				toolbar.add(b);
			}

			this.add(toolbar, BorderLayout.PAGE_START);
			
			this.currentState = new IdleState();
			
			JButton select = new JButton("Selektiraj");
			select.addActionListener(new ActionListener() {
				@Override
				public void actionPerformed(ActionEvent e) {
					currentState = new SelectShapeState(model);
				}
			});
			select.setFocusable(false);
			toolbar.add(select);
			
			JButton delete = new JButton("Izbriši");
			delete.addActionListener(new ActionListener() {
				@Override
				public void actionPerformed(ActionEvent e) {
					currentState = new EraserState(model);
				}
			});
			delete.setFocusable(false);
			toolbar.add(delete);
			
			JButton svgExport = new JButton("SVG export");
			svgExport.addActionListener(new ActionListener() {
				@Override
				public void actionPerformed(ActionEvent e) {
					
					JFileChooser jfc = new JFileChooser();
					int res = jfc.showOpenDialog(jfc.getParent());
					
					if (res == JFileChooser.APPROVE_OPTION) {
						SVGRendererImpl r = new SVGRendererImpl(jfc.getSelectedFile().getAbsolutePath());
						for(GraphicalObject o : model.list())
							o.render(r);
						try {r.close(); } 
						catch (IOException e1) { }
						
						JOptionPane.showMessageDialog(jfc.getParent(), "SVG file was saved.");
					} else if (res == JFileChooser.CANCEL_OPTION) { 
						JOptionPane.showMessageDialog(jfc.getParent(), "Nothing was saved.");
					}
					
				}
			});
			svgExport.setFocusable(false);
			toolbar.add(svgExport);
			
			JButton save = new JButton("Pohrani");
			save.addActionListener(new ActionListener() {
				@Override
				public void actionPerformed(ActionEvent e) {

					JFileChooser jfc = new JFileChooser();
					int res = jfc.showOpenDialog(jfc.getParent());
					
					if (res == JFileChooser.APPROVE_OPTION) {
					
						List<String> rows = new ArrayList<>();
						
						for(GraphicalObject o : model.list()) 
							o.save(rows);
						
						try {
							Files.write(Paths.get(jfc.getSelectedFile().getAbsolutePath()), rows, Charset.defaultCharset());
						} catch (IOException e1) { }
						
						JOptionPane.showMessageDialog(jfc.getParent(), "Native file was saved.");
					
					} else if (res == JFileChooser.CANCEL_OPTION) { 
						JOptionPane.showMessageDialog(jfc.getParent(), "Nothing was loaded.");
					}
				
				}
			});
			save.setFocusable(false);
			toolbar.add(save);
			
			
			JButton load = new JButton("Učitaj");
			load.addActionListener(new ActionListener() {
				@Override
				public void actionPerformed(ActionEvent e) {

					JFileChooser jfc = new JFileChooser();
					int res = jfc.showOpenDialog(jfc.getParent());
					
					if (res == JFileChooser.APPROVE_OPTION) {
						
						Stack<GraphicalObject> stack = new Stack<>();
						List<String> rows = new ArrayList<>();
						
						try {
							rows = Files.readAllLines(Paths.get(jfc.getSelectedFile().getAbsolutePath()), Charset.defaultCharset());
							for(String s : rows) {
								String[] arr =  s.split(" ");
								String data = String.join(" ", Arrays.copyOfRange(arr, 1, arr.length));
								switch (arr[0]) {
								case "@LINE": new LineSegment().load(stack, data);
									break;
								case "@OVAL": new Oval().load(stack, data);
									break;
								case "@COMP": new CompositeShape().load(stack, data);
									break;
								}
							}
							stack.forEach(o -> model.addGraphicalObject(o));
						} catch (IOException e1) { }
						
						JOptionPane.showMessageDialog(jfc.getParent(), "Native file was loaded.");
					} else if (res == JFileChooser.CANCEL_OPTION) { 
						JOptionPane.showMessageDialog(jfc.getParent(), "Nothing was loaded.");
					}
				
				}
			});
			load.setFocusable(false);
			toolbar.add(load);

			
			this.addMouseListener(new MouseAdapter() {
				@Override
				public void mouseClicked(MouseEvent e) {
					currentState.mouseDown(new Point(e.getX(),  e.getY()), e.isShiftDown(), e.isControlDown());
					e.consume();
				}
				
				@Override
				public void mouseReleased(MouseEvent e) {
					currentState.mouseUp(new Point(e.getX(),  e.getY()), e.isShiftDown(), e.isControlDown());
					e.consume();
				}
				
			});
			
			this.addMouseMotionListener(new MouseMotionAdapter() {
				@Override
				public void mouseDragged(MouseEvent e) {
					currentState.mouseDragged(new Point(e.getX(), e.getY()));
					e.consume();
				}
			});

		}

		private KeyListener kl = new KeyAdapter() {
			@Override
			public void keyPressed(KeyEvent e) {
				if (e.getKeyCode() == KeyEvent.VK_ESCAPE) {
					currentState.onLeaving();
					currentState = new IdleState();
				} else { currentState.keyPressed(e.getKeyCode()); }
				e.consume();
			}
		};
		
		
		@Override
		public void paintComponent(Graphics g) {
			Graphics2D g2d = (Graphics2D)g;
			Renderer r = new G2DRendererImpl(g2d);
			for(GraphicalObject o : model.list()) {
				o.render(r);
				currentState.afterDraw(r,  o);
			}
			currentState.afterDraw(r);
		}
	}
}

