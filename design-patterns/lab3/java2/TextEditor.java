package MyTextEditor;

import java.awt.Color;
import java.awt.Font;
import java.awt.FontMetrics;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Insets;
import java.awt.event.ActionEvent;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Iterator;

import javax.swing.AbstractAction;
import javax.swing.Action;
import javax.swing.JComponent;
import javax.swing.KeyStroke;

public class TextEditor extends JComponent {

	private TextEditorModel model;
	
	public TextEditor(String s) {
		this.model = new TextEditorModel(s);
		this.model.setCursorLocation(new Location(0, 0));
		this.model.addCursorObserver((loc) -> repaint());
		this.model.addTextObserver(() -> repaint());

		addToMaps("BACK_SPACE", backspace);
		addToMaps("DELETE", delete);
		addToMaps("UP", up);
		addToMaps("DOWN", down);
		addToMaps("LEFT", left);
		addToMaps("RIGHT", right);

		for(char c=48; c<116; c++)
			addToMaps(String.valueOf(c), insertAction);

		addToMaps("SPACE", insertAction);
		addToMaps("ENTER", insertAction);
		addToMaps("shift LEFT", shiftLeft);
		addToMaps("shift RIGHT", shiftRight);
		addToMaps("shift UP", shiftUp);
		addToMaps("shift DOWN", shiftDown);
		addToMaps("ctrl C", copyAction);
		addToMaps("ctrl V", pasteAction);
		addToMaps("ctrl X", cropAction);
		addToMaps("control shift V", pasteAction2);
		
		this.requestFocusInWindow();

		addToMaps("control Z", undoAction);
		addToMaps("control Y", redoAction);
	}


	public void addToMaps(String keyStroke, Action action) {
		this.getInputMap().put(KeyStroke.getKeyStroke(keyStroke), keyStroke);
		this.getActionMap().put(keyStroke, action);
	}

	@Override
	public void paintComponent(Graphics g) {
		Graphics2D g2 = (Graphics2D) g;
		g.setFont(Font.decode(Font.MONOSPACED+" "+12));

		FontMetrics fm = g2.getFontMetrics();
		Insets ins = this.getInsets();
		int left = ins.left;
		int top = ins.top;

		int i=0;
		Iterator<String> it = this.model.allLines();
		Location c = this.model.getCursorLocation();

		LocationRange selected = this.model.getSelectionRange();
		Location start = null, end = null;
		boolean selectedBool = selected != null;
		if(selectedBool)  {
			start = selected.sort().getStart();
			end = selected.sort().getEnd();
		}

		int y = c.getY() > 0 ? c.getY() : 0;
		String cursorString = "";

		while(it.hasNext()) {
			String l = it.next();

			if (i == y) 
				cursorString = new String(l);

			if(selectedBool && i >= start.getY() && i <= end.getY())  {

				String temp = null;
				if(start.getY() == i && end.getY() == i)
					temp = l.substring(start.getX(), end.getX());
				else if (start.getY() == i)
					temp = l.substring(start.getX());
				else if (end.getY() == i)
					temp = l.substring(0, end.getX());
				else if (i >= start.getY() && i <= end.getY())
					temp = l;
				else { }

				if(temp != null) {
					g2.setColor(Color.yellow);
					g2.fillRect(left + i == start.getY() ? fm.stringWidth(l.substring(0, start.getX())) : 0,
							fm.getHeight()*i + fm.getDescent(),
							fm.stringWidth(temp), 
							fm.getHeight());
					g2.setColor(Color.black);
				}
			}

			g2.drawString(l, left, top+fm.getHeight()*(i+1));
			i++;
		}

		int x = cursorString.length() > c.getX() ? c.getX() : cursorString.length();
		int offset = 5;
		g2.drawLine(left + fm.stringWidth(cursorString.substring(0, x == -1 ? 0 : x)), 
				top + fm.getHeight()*(y) + offset,
				left + fm.stringWidth(cursorString.substring(0, x == -1 ? 0 : x)), 
				top + fm.getHeight()*(y+1) + offset);
	}
	
	Action cursorToDocumentStartAction = new AbstractAction() {
		@Override
		public void actionPerformed(ActionEvent e) {
			model.cursorToDocumentStart();
		}
	};

	Action cursorToDocumentEndAction = new AbstractAction() {
		@Override
		public void actionPerformed(ActionEvent e) {
			model.cursorToDocumentEnd();
		}
	};

	Action clearDocumentAction = new AbstractAction() {
		@Override
		public void actionPerformed(ActionEvent e) {
			model.clearDocument();
		}
	};

	Action deleteSelectionAction = new AbstractAction() {
		@Override
		public void actionPerformed(ActionEvent e) {
			model.deleteSelected();
		}
	};


	private Path filePath;
	public void setFilePath(Path filePath) {
		this.filePath = filePath;
	}

	public Path getFilePath() {
		return this.filePath;
	}

	public void open(){
		try { model.readFromPath(filePath); }
		catch (IOException e1) { }
		setFilePath(null);
	};

	public void save() {
		try { model.saveLines(filePath); }
		catch (IOException e1) { }
		setFilePath(null);

	};
	
	public TextEditorModel getModel() {
		return this.model;
	}

	public void exit() {
		System.exit(0);
	};

	Action undoAction = new AbstractAction() {
		@Override
		public void actionPerformed(ActionEvent e) {
			model.undo();
		}
	};

	Action redoAction = new AbstractAction() {
		@Override
		public void actionPerformed(ActionEvent e) {
			model.redo();
		}
	};

	Action copyAction = new AbstractAction() {
		@Override
		public void actionPerformed(ActionEvent e) {
			model.copy();
		}
	};

	Action pasteAction = new AbstractAction() {
		@Override
		public void actionPerformed(ActionEvent e) {
			model.paste();
		}
	};

	Action cropAction = new AbstractAction() {
		@Override
		public void actionPerformed(ActionEvent e) {
			model.crop();
		}
	};

	Action pasteAction2 = new AbstractAction() {
		@Override
		public void actionPerformed(ActionEvent e) {
			model.pasteCtrl();
		}
	};

	Action insertAction = new AbstractAction() {
		@Override
		public void actionPerformed(ActionEvent e) {
			model.insert(e.getActionCommand().toCharArray()[0]);
		}
	};


	Action shiftLeft = new AbstractAction() {
		@Override
		public void actionPerformed(ActionEvent e) {
			model.moveCursorShift(() -> model.moveLeft());
		}
	};

	Action shiftRight = new AbstractAction() {
		@Override
		public void actionPerformed(ActionEvent e) {
			model.moveCursorShift(() -> model.moveRight());
		}
	};

	Action shiftDown = new AbstractAction() {
		@Override
		public void actionPerformed(ActionEvent e) {
			model.moveCursorShift(() -> model.moveDown());
		}
	};

	Action shiftUp = new AbstractAction() {
		@Override
		public void actionPerformed(ActionEvent e) {
			model.moveCursorShift(() -> model.moveUp());
		}
	};

	Action backspace = new AbstractAction() {
		@Override
		public void actionPerformed(ActionEvent e) {
			model.moveCursorResetSelection(() -> model.deleteBefore());
		}
	};

	Action delete = new AbstractAction() {
		@Override
		public void actionPerformed(ActionEvent e) {
			model.moveCursorResetSelection(() -> model.deleteAfter());
		}
	};

	Action up = new AbstractAction() {
		@Override
		public void actionPerformed(ActionEvent e) {
			model.moveCursorResetSelection(() -> model.moveUp());
		}
	}; 

	Action down = new AbstractAction() {
		@Override
		public void actionPerformed(ActionEvent e) {
			model.moveCursorResetSelection(() -> model.moveDown());
		}
	}; 

	Action left = new AbstractAction() {
		@Override
		public void actionPerformed(ActionEvent e) {
			model.moveCursorResetSelection(() -> model.moveLeft());
		}
	};

	Action right = new AbstractAction() {
		@Override
		public void actionPerformed(ActionEvent e) {
			model.moveCursorResetSelection(() -> model.moveRight());
		}
	};


}



