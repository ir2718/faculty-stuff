package MyTextEditor;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Container;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;

import javax.swing.AbstractAction;
import javax.swing.Action;
import javax.swing.BorderFactory;
import javax.swing.JButton;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JMenu;
import javax.swing.JMenuBar;
import javax.swing.JMenuItem;
import javax.swing.JOptionPane;
import javax.swing.JToolBar;
import javax.swing.KeyStroke;
import javax.swing.SwingConstants;
import javax.swing.SwingUtilities;
import javax.swing.WindowConstants;

public class MainFrame extends JFrame {

	private TextEditor te;

	public static void main(String[] args) {
		String s = "Ovo \n je neki \n dugacki tekst \n s vise redaka";
		SwingUtilities.invokeLater(new Runnable() {
			@Override
			public void run() {
				MainFrame mf = null;
				try { mf = new MainFrame(s); }
				catch (Exception e) { }
				mf.setVisible(true);
			}
		});
	}

	public MainFrame(String s) throws Exception {
		setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
		setLocation(0, 0);
		setSize(300, 300);

		this.te = new TextEditor(s);

		Container cp = this.getContentPane();
		cp.setLayout(new BorderLayout());
		cp.add(te, BorderLayout.CENTER);

		createActions(te);
		createMenusAndToolbar(te);
		createStatusBar(te);
	}

	private void createActions(TextEditor te) {
		openAction.putValue(Action.ACCELERATOR_KEY, KeyStroke.getKeyStroke("control O"));
		saveAction.putValue(Action.ACCELERATOR_KEY, KeyStroke.getKeyStroke("control S")); 
		exitAction.putValue(Action.ACCELERATOR_KEY, KeyStroke.getKeyStroke("control E")); 

		te.undoAction.putValue(Action.ACCELERATOR_KEY, KeyStroke.getKeyStroke("control Z"));
		te.redoAction.putValue(Action.ACCELERATOR_KEY, KeyStroke.getKeyStroke("control Y")); 
		te.cropAction.putValue(Action.ACCELERATOR_KEY, KeyStroke.getKeyStroke("control X")); 
		te.copyAction.putValue(Action.ACCELERATOR_KEY, KeyStroke.getKeyStroke("control C"));
		te.pasteAction.putValue(Action.ACCELERATOR_KEY, KeyStroke.getKeyStroke("control V")); 
		te.pasteAction2.putValue(Action.ACCELERATOR_KEY, KeyStroke.getKeyStroke("control shift V"));
		te.deleteSelectionAction.putValue(Action.ACCELERATOR_KEY, KeyStroke.getKeyStroke("control D"));
		te.clearDocumentAction.putValue(Action.ACCELERATOR_KEY, KeyStroke.getKeyStroke("control R"));

		te.cursorToDocumentStartAction.putValue(Action.ACCELERATOR_KEY, KeyStroke.getKeyStroke("control 0"));
		te.cursorToDocumentEndAction.putValue(Action.ACCELERATOR_KEY, KeyStroke.getKeyStroke("control 9"));
	}

	private void createMenusAndToolbar(TextEditor te) throws Exception {

		JToolBar toolBar = new JToolBar();
		toolBar.setFloatable(true);

		JButton b1 = getButtonForToolbar("Undo", te.undoAction);
		toolBar.add(b1);
		JButton b2 = getButtonForToolbar("Redo", te.redoAction);
		toolBar.add(b2);
		toolBar.addSeparator();

		JButton b3 = getButtonForToolbar("Cut", te.cropAction);
		toolBar.add(b3);
		JButton b4 = getButtonForToolbar("Copy", te.copyAction);
		toolBar.add(b4);
		JButton b5 = getButtonForToolbar("Paste", te.pasteAction);
		toolBar.add(b5);
		
		Arrays.stream(toolBar.getComponents()).forEach(c -> {
			c.setFocusable(false);
			c.setEnabled(false);
		});
		
		toolBar.setAlignmentY(SwingConstants.VERTICAL);

		this.getContentPane().add(toolBar, BorderLayout.PAGE_START);

		JMenuBar menuBar = new JMenuBar();

		JMenu fileMenu = new JMenu("File");
		menuBar.add(fileMenu);

		fileMenu.add(getJMenuItemForJMenu("Open", openAction));
		fileMenu.add(getJMenuItemForJMenu("Save", saveAction));
		fileMenu.add(getJMenuItemForJMenu("Exit", exitAction));

		JMenu editMenu = new JMenu("Edit");
		menuBar.add(editMenu);

		JMenuItem undo = getJMenuItemForJMenu("Undo", te.undoAction);
		SelectionObserver undoObserver = new SelectionObserver() {
			@Override
			public void update() { 
				if (UndoManager.getInstance().getUndoStack().isEmpty()) {
					undo.setEnabled(false);
					b1.setEnabled(false);
				} else {
					undo.setEnabled(true);
					b1.setEnabled(true);
				}
			}
		};
		undo.setEnabled(false);
		te.getModel().addSelectionObserver(undoObserver);
		editMenu.add(undo);

		JMenuItem redo = getJMenuItemForJMenu("Redo", te.redoAction);
		SelectionObserver redoObserver = new SelectionObserver() {
			@Override
			public void update() { 
				if (UndoManager.getInstance().getRedoStack().isEmpty()) { 
					redo.setEnabled(false);
					b2.setEnabled(false);
				} else {
					redo.setEnabled(true);
					b2.setEnabled(true);
				}
			}
		};
		redo.setEnabled(false);
		te.getModel().addSelectionObserver(redoObserver);
		editMenu.add(redo);


		JMenuItem crop = getJMenuItemForJMenu("Cut", te.cropAction);
		SelectionObserver cropObserver = new SelectionObserver() {
			@Override
			public void update() { 
				if (te.getModel().getSelectionRange() == null) {
					crop.setEnabled(false);
					b3.setEnabled(false);
				} else {
					crop.setEnabled(true);
					b3.setEnabled(true);
				}
			}
		};
		crop.setEnabled(false);
		te.getModel().addSelectionObserver(cropObserver);
		editMenu.add(crop);

		JMenuItem copy = getJMenuItemForJMenu("Copy", te.copyAction);
		SelectionObserver copyObserver = new SelectionObserver() {
			@Override
			public void update() { 
				if (te.getModel().getSelectionRange() == null) {
					copy.setEnabled(false);
					b4.setEnabled(false);
				} else {
					copy.setEnabled(true);
					b4.setEnabled(true);
				}
			}
		};
		copy.setEnabled(false);
		te.getModel().addSelectionObserver(copyObserver);
		editMenu.add(copy);

		JMenuItem paste = getJMenuItemForJMenu("Paste", te.pasteAction);
		SelectionObserver pasteObserver = new SelectionObserver() {
			@Override
			public void update() { 
				if (te.getModel().getClipboardStack().isEmpty()) {
					paste.setEnabled(false);
					b5.setEnabled(false);
				} else {
					paste.setEnabled(true);
					b5.setEnabled(true);
				}
			}
		};
		paste.setEnabled(false);
		te.getModel().addSelectionObserver(pasteObserver);
		editMenu.add(paste);

		JMenuItem paste2 = getJMenuItemForJMenu("Paste and take", te.pasteAction2);
		SelectionObserver pasteObserver2 = new SelectionObserver() {
			@Override
			public void update() { 
				if (te.getModel().getClipboardStack().isEmpty()) paste2.setEnabled(false);
				else paste2.setEnabled(true);
			}
		};
		paste2.setEnabled(false);
		te.getModel().addSelectionObserver(pasteObserver2);
		editMenu.add(paste2);

		editMenu.add(getJMenuItemForJMenu("Delete selected", te.deleteSelectionAction));
		editMenu.add(getJMenuItemForJMenu("Clear document", te.clearDocumentAction));

		JMenu moveMenu = new JMenu("Move");
		menuBar.add(moveMenu);

		moveMenu.add(getJMenuItemForJMenu("Cursor to document start", te.cursorToDocumentStartAction));
		moveMenu.add(getJMenuItemForJMenu("Cursor to document end", te.cursorToDocumentEndAction));

		JMenu pluginMenu = new JMenu("Plugins");
		menuBar.add(pluginMenu);

		String pathOfClass = "./src/MyTextEditor/plugins";
		File currentFolder = new File(pathOfClass);
		String[] files = currentFolder.list();
		for(String s : files) {
			String newPlugin = s.split("\\.")[0];
			Plugin p = PluginFactory.getInstance(newPlugin, "MyTextEditor.plugins.");
			JMenuItem i = new JMenuItem();
			i.setText(p.getName());
			i.setToolTipText(p.getDescription());
			i.addActionListener((e) -> p.execute(te.getModel(), UndoManager.getInstance(), te.getModel().getClipboardStack()));
			pluginMenu.add(i);
		}

		this.setJMenuBar(menuBar);
	}

	private static JButton getButtonForToolbar(String name, Action a) {
		JButton b = new JButton(name);
		b.addActionListener(a);
		return b;
	}

	private static JMenuItem getJMenuItemForJMenu(String name, Action a) {
		JMenuItem i = new JMenuItem(name);
		i.addActionListener(a);
		return i;
	}

	private void createStatusBar(TextEditor te) {
		JLabel length = new JLabel();
		length.setText("Length: " + te.getModel().getLines().size());

		JLabel location = new JLabel();
		location.setText("Ln: 0, Col: 0");

		JToolBar statusBar = new JToolBar();
		statusBar.setLayout(new GridLayout(1, 0));
		statusBar.add(length);
		statusBar.add(location);
		statusBar.setSize(this.getWidth(), 50);
		statusBar.setBorder(BorderFactory.createMatteBorder(1, 0, 0, 0, Color.BLACK));

		te.getModel().addTextObserver(new TextObserver() {
			@Override
			public void updateText() {
				length.setText("Length: " + te.getModel().getLines().size());
			}
		});

		te.getModel().addCursorObserver(new CursorObserver() {
			@Override
			public void updateCursorLocation(Location loc) {
				Location l = te.getModel().getCursorLocation();
				location.setText("Ln: "+ l.getY() +", Col: " + l.getX());
			}
		});

		this.getContentPane().add(statusBar, BorderLayout.SOUTH);
	}

	Action openAction = new AbstractAction() {
		@Override
		public void actionPerformed(ActionEvent e) {
			JFileChooser fc = new JFileChooser();
			fc.setDialogTitle("Open file");
			if(fc.showOpenDialog(MainFrame.this)!=JFileChooser.APPROVE_OPTION) {
				return;
			}
			File fileName = fc.getSelectedFile();
			Path filePath = fileName.toPath();
			if(!Files.isReadable(filePath)) {
				JOptionPane.showMessageDialog(
						MainFrame.this, 
						"The file "+fileName.getAbsolutePath()+" doesn't exist.", 
						"Error.", 
						JOptionPane.ERROR_MESSAGE);
				return;
			}

			try {
				te.setFilePath(filePath);
				te.open();
			} catch (Exception exc) {
				JOptionPane.showMessageDialog(
						MainFrame.this, 
						"Error while loading document "+fileName.getAbsolutePath()+".", 
						"Error.", 
						JOptionPane.ERROR_MESSAGE);
			}
		}
	};


	private Action saveAction = new AbstractAction() {

		@Override
		public void actionPerformed(ActionEvent e) {
			if(te.getFilePath()==null ) {
				saveAsDocumentAction.actionPerformed(e);
				return;
			}

			try {
				te.save();
			} catch(Exception exc) {
				JOptionPane.showMessageDialog(
						MainFrame.this, 
						"Error while saving document "+te.getFilePath()+".", 
						"Error.", 
						JOptionPane.ERROR_MESSAGE);
				return;
			}

			JOptionPane.showMessageDialog(
					MainFrame.this, 
					"File is saved.", 
					"Information", 
					JOptionPane.INFORMATION_MESSAGE);


		}
	};

	private Action saveAsDocumentAction = new AbstractAction() {

		@Override
		public void actionPerformed(ActionEvent e) {

			JFileChooser jfc = new JFileChooser();
			jfc.setDialogTitle("Save as document");
			if(jfc.showSaveDialog(MainFrame.this)!=JFileChooser.APPROVE_OPTION) {
				JOptionPane.showMessageDialog(
						MainFrame.this, 
						"Nothing was recorded..", 
						"Warning", 
						JOptionPane.WARNING_MESSAGE);
				return;
			}

			if( jfc.getSelectedFile().exists() ) {
				int input = JOptionPane.showConfirmDialog(MainFrame.this, 
						"The file already exists. Do you want to overwrite it?", 
						"Warning", 
						JOptionPane.YES_NO_CANCEL_OPTION);
				if( input != JOptionPane.YES_OPTION ) 
					return;
			}

			Path p = jfc.getSelectedFile().toPath();

			try {
				te.setFilePath(p);
				te.save();
			} catch (Exception exc) {
				JOptionPane.showMessageDialog(
						MainFrame.this, 
						"Error while saving document as "+p+".", 
						"Error.", 
						JOptionPane.ERROR_MESSAGE);
				return;
			}

			JOptionPane.showMessageDialog(
					MainFrame.this, 
					"File is saved.", 
					"Information", 
					JOptionPane.INFORMATION_MESSAGE);
		}
	};

	private Action exitAction = new AbstractAction() {

		@Override
		public void actionPerformed(ActionEvent e) {
			te.exit();
		}
	};
}
