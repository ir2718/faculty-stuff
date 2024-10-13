package hr.fer.oprpp1.hw08.jnotepadpp;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.KeyEvent;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.text.Collator;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Date;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.stream.Collectors;

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
import javax.swing.JTextArea;
import javax.swing.JToolBar;
import javax.swing.KeyStroke;
import javax.swing.SwingUtilities;
import javax.swing.Timer;
import javax.swing.WindowConstants;
import javax.swing.event.CaretEvent;
import javax.swing.event.CaretListener;
import javax.swing.text.BadLocationException;
import javax.swing.text.Caret;
import javax.swing.text.Document;
import javax.swing.text.Element;
import javax.swing.text.JTextComponent;

import hr.fer.oprpp1.hw08.local.LocalizableAction;
import hr.fer.oprpp1.hw08.local.LocalizationProvider;
import hr.fer.oprpp1.hw08.local.FormLocalizationProvider;
import hr.fer.oprpp1.hw08.local.LJLabel;
import hr.fer.oprpp1.hw08.local.LJMenu;

import static java.lang.Math.*;

/**
 * The main programme class for demonstrating the functionalities of
 * the JNotepad++.
 * 
 * @author Ivan Rep
 */
public class JNotepadPP extends JFrame {

	private static final long serialVersionUID = 1L;

	private static final String UNNAMED_DOC = "(unnamed)";

	private DefaultMultipleDocumentModel m;
	private JToolBar statusBar;
	private CaretListener listener;
	private JMenu caseMenu;
	private FormLocalizationProvider flp = new FormLocalizationProvider(LocalizationProvider.getInstance(), this);

	/**
	 * Constructor.
	 */
	public JNotepadPP() {

		setDefaultCloseOperation(WindowConstants.DO_NOTHING_ON_CLOSE);
		this.addWindowListener(new WindowAdapter() {
			@Override
			public void windowClosing(WindowEvent e) {
				checkAllSaved();
				JNotepadPP.this.dispose();
			}
		});

		setLocation(0, 0);
		setSize(1000, 800);
		setTitle("JNotepad++");

		initGUI();
	}


	/**
	 * Initializes the GUI and listeners.
	 * 
	 * @see createActions
	 * @see createMenus
	 * @see createToolbars
	 * @see createStatusBar
	 */
	private void initGUI() {

		m = new DefaultMultipleDocumentModel();

		this.getContentPane().setLayout(new BorderLayout());
		this.getContentPane().add( m, BorderLayout.CENTER);
		m.addMultipleDocumentListener(new MultipleDocumentListener() {

			@Override
			public void currentDocumentChanged(SingleDocumentModel previousModel, SingleDocumentModel currentModel) {
				String afterFilePath = " - JNotepad++";
				if(currentModel == null ) {
					setTitle("JNotepad++");
				} else {
					setTitle( (currentModel.getFilePath() == null ? 
							UNNAMED_DOC : currentModel.getFilePath().toString() ) + afterFilePath );
					listener.caretUpdate(null);
				}
			}

			@Override
			public void documentAdded(SingleDocumentModel model) { 
				model.getTextComponent().addCaretListener(listener); 
			}

			@Override
			public void documentRemoved(SingleDocumentModel model) {
				model.getTextComponent().removeCaretListener(listener);
			}

		});

		createStatusBar();
		this.getContentPane().add( statusBar, BorderLayout.SOUTH);

		createActions();
		createMenus();
		createToolbars();

	}

	/**
	 * Initializes the status bar, the clock and creates caret listener.
	 */
	private void createStatusBar() {
		JLabel length = new LJLabel("length", flp);
		length.setText( length.getText()+":" );
		JLabel other = new JLabel("Ln: Col: Sel:");
		this.statusBar = new JToolBar();
		this.statusBar.setLayout(new GridLayout(1, 0));
		this.statusBar.add(length);
		this.statusBar.add(other);
		this.statusBar.setSize(this.getWidth(), 50);
		this.statusBar.setBorder(BorderFactory.createMatteBorder(1, 0, 0, 0, Color.BLACK));

		int tickTime = 1000;
		SimpleDateFormat sdf = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
		JLabel clock = new JLabel( sdf.format(new Date() ) );
		Timer t = new Timer(tickTime, (e) -> clock.setText( sdf.format(new Date() ) ) );
		t.setRepeats(true);
		t.start();
		this.statusBar.add(clock);

		listener = new CaretListener() {
			@Override
			public void caretUpdate(CaretEvent e) {


				if( m.getNumberOfDocuments()!=0 ) {
					int len, ln = 0, col = 0, sel = 0;
					JTextArea currArea = m.getCurrentDocument().getTextComponent();
					Caret currCaret = currArea.getCaret();
					len  = currArea.getText().length();
					sel = abs( currCaret.getMark() - currCaret.getDot() );

					if(sel==0) JNotepadPP.this.caseMenu.setEnabled(false);
					else JNotepadPP.this.caseMenu.setEnabled(true);

					try {
						ln = currArea.getLineOfOffset( currArea.getCaretPosition() ) + 1; 
						col = currCaret.getDot()-currArea.getLineStartOffset(ln-1);
					} catch (BadLocationException e1) { }

					length.setText( String.format("%s: %d", flp.getString("length"), len ) );
					other.setText( "Ln: "+ln+" Col: "+col+" Sel: "+sel );
					return;
				} else {
					JNotepadPP.this.caseMenu.setEnabled(false);
				}


				length.setText( String.format("%s:", flp.getString("length") ) );
				other.setText( "Ln: Col: Sel: " );

			}	
		};

		flp.addLocalizationListener( () -> listener.caretUpdate(null) );

	}

	// Actions

	/**
	 * Action opens a blank document. 
	 */
	private Action createBlankDocumentAction = new LocalizableAction( "new" , flp ) {

		private static final long serialVersionUID = 1L;

		@Override
		public void actionPerformed(ActionEvent e) {
			m.createNewDocument();
		}
	};


	/**
	 * Action opens a document from its' path. 
	 */
	private Action openDocumentAction = new LocalizableAction( "open" , flp ) {

		private static final long serialVersionUID = 1L;

		@Override
		public void actionPerformed(ActionEvent e) {
			JFileChooser fc = new JFileChooser();
			fc.setDialogTitle("Open file");
			if(fc.showOpenDialog(JNotepadPP.this)!=JFileChooser.APPROVE_OPTION) {
				return;
			}
			File fileName = fc.getSelectedFile();
			Path filePath = fileName.toPath();
			if(!Files.isReadable(filePath)) {
				JOptionPane.showMessageDialog(
						JNotepadPP.this, 
						"The file "+fileName.getAbsolutePath()+" doesn't exist.", 
						"Error.", 
						JOptionPane.ERROR_MESSAGE);
				return;
			}

			try {
				JNotepadPP.this.m.loadDocument(filePath);
			} catch (Exception exc) {
				JOptionPane.showMessageDialog(
						JNotepadPP.this, 
						"Error while loading document "+fileName.getAbsolutePath()+".", 
						"Error.", 
						JOptionPane.ERROR_MESSAGE);
			}

		}
	};

	/**
	 * Action saves a document to the current documents' path.
	 * If path is null, the user can choose where the document will be saved. 
	 */
	private Action saveDocumentAction = new LocalizableAction( "save" , flp ) {

		private static final long serialVersionUID = 1L;

		@Override
		public void actionPerformed(ActionEvent e) {
			if( m.getCurrentDocument().getFilePath()==null ) {
				saveAsDocumentAction.actionPerformed(e);
				return;
			}

			SingleDocumentModel curr = m.getCurrentDocument();

			try {
				m.saveDocument( curr, curr.getFilePath() );
			} catch(Exception exc) {
				JOptionPane.showMessageDialog(
						JNotepadPP.this, 
						"Error while saving document "+curr.getFilePath()+".", 
						"Error.", 
						JOptionPane.ERROR_MESSAGE);
				return;
			}

			JOptionPane.showMessageDialog(
					JNotepadPP.this, 
					"File is saved.", 
					"Information", 
					JOptionPane.INFORMATION_MESSAGE);


		}
	};

	/**
	 * Helper method checks if all the document in the model are saved.
	 */
	private void checkAllSaved() {

		for(int i=0; i<m.getNumberOfDocuments(); i++)
			checkOneSaved(m.getDocument(i));

	}

	/**
	 * Helper method checks if one model is saved.
	 * If it isn't asks the user if he wants to save.
	 * 
	 * @param model the model that is checked
	 */
	private void checkOneSaved(SingleDocumentModel model) {

		if( !model.isModified() ) return;

		boolean isUnnamed = model.getFilePath()==null;
		int input = JOptionPane.showConfirmDialog(JNotepadPP.this, 
				isUnnamed ? "An unnamed file is not saved. Do you want to save it?" :
					"The file "+ model.getFilePath()+". Do you want to save it?", 
					"Warning", 
					JOptionPane.YES_NO_CANCEL_OPTION);

		if( input == JOptionPane.YES_OPTION ) {
			if( isUnnamed ) this.saveAsDocumentAction.actionPerformed(null);
			else this.saveDocumentAction.actionPerformed(null);
			return;
		}

		if( input == JOptionPane.NO_OPTION ) {
			m.closeDocument(m.getCurrentDocument());
			return;
		}

		return;


	}

	/**
	 * Action saves a document to the path the user chooses. 
	 * If a file with the given path already exists, asks the user if he wants to overwrite. 
	 */
	private Action saveAsDocumentAction = new LocalizableAction( "saveAs" , flp ) {

		private static final long serialVersionUID = 1L;

		@Override
		public void actionPerformed(ActionEvent e) {

			JFileChooser jfc = new JFileChooser();
			jfc.setDialogTitle("Save as document");
			if(jfc.showSaveDialog(JNotepadPP.this)!=JFileChooser.APPROVE_OPTION) {
				JOptionPane.showMessageDialog(
						JNotepadPP.this, 
						"Nothing was recorded..", 
						"Warning", 
						JOptionPane.WARNING_MESSAGE);
				return;
			}

			if( jfc.getSelectedFile().exists() ) {
				int input = JOptionPane.showConfirmDialog(JNotepadPP.this, 
						"The file already exists. Do you want to overwrite it?", 
						"Warning", 
						JOptionPane.YES_NO_CANCEL_OPTION);
				if( input != JOptionPane.YES_OPTION ) 
					return;
			}

			Path p = jfc.getSelectedFile().toPath();

			try {
				m.saveDocument(m.getCurrentDocument(), p);
			} catch (Exception exc) {
				JOptionPane.showMessageDialog(
						JNotepadPP.this, 
						"Error while saving document as "+p+".", 
						"Error.", 
						JOptionPane.ERROR_MESSAGE);
				return;
			}

			JOptionPane.showMessageDialog(
					JNotepadPP.this, 
					"File is saved.", 
					"Information", 
					JOptionPane.INFORMATION_MESSAGE);
		}
	};

	/**
	 * Closes the current document and updates the listeners.
	 */
	private Action closeDocumentAction = new LocalizableAction( "close" , flp ) {

		private static final long serialVersionUID = 1L;

		@Override
		public void actionPerformed(ActionEvent e) {
			JNotepadPP app = JNotepadPP.this;
			SingleDocumentModel curr = app.m.getCurrentDocument();

			if(!curr.isModified()) app.m.closeDocument(curr);
			else checkOneSaved(curr);

			listener.caretUpdate(null);

		}
	};

	/**
	 * Action used for getting information about the current document.
	 * Shows the user how many non blank character, lines and characters his document has.
	 */
	private Action infoAction = new LocalizableAction( "info" , flp ) {

		private static final long serialVersionUID = 1L;

		@Override
		public void actionPerformed(ActionEvent e) {

			JTextArea area = m.getCurrentDocument().getTextComponent();

			char[] charText = area.getText().toCharArray();
			int countNonBlank = 0;
			for( char c : charText ) 
				if( !Character.isWhitespace(c) ) countNonBlank++;


			JOptionPane.showMessageDialog(JNotepadPP.this, 
					"Your document has "+area.getText().length()+" characters, "
							+countNonBlank+" non-blank characters and "+area.getLineCount()+" lines." );
		}
	};

	/**
	 * Cuts the selected text and saves it to the clipboard.
	 */
	private Action cutAction = new LocalizableAction( "cut" , flp ) {

		private static final long serialVersionUID = 1L;

		@Override
		public void actionPerformed(ActionEvent e) { 
			JNotepadPP.this.m.getCurrentDocument().getTextComponent().cut();
		}
	};

	/**
	 * Copies the selected text and saves it to the clipboard.
	 */
	private Action copyAction = new LocalizableAction( "copy" , flp ) {

		private static final long serialVersionUID = 1L;

		@Override
		public void actionPerformed(ActionEvent e) { 
			JNotepadPP.this.m.getCurrentDocument().getTextComponent().copy();
		}
	};

	/**
	 * Pastes the current text from the clipboard into the document.
	 */
	private Action pasteAction = new LocalizableAction( "paste" , flp ) {

		private static final long serialVersionUID = 1L;

		@Override
		public void actionPerformed(ActionEvent e) {
			JNotepadPP.this.m.getCurrentDocument().getTextComponent().paste();
		}
	};

	/**
	 * Exits the application.
	 * If there are unsaved documents, asks the user whether he wants to save.
	 */
	private Action exitAction = new LocalizableAction( "exit" , flp ) {

		private static final long serialVersionUID = 1L;

		@Override
		public void actionPerformed(ActionEvent e) {
			checkAllSaved();
		}
	};

	/**
	 * Interface used for strategy design pattern.
	 * Reads the selected text and changes the characters.
	 * 
	 * @author Ivan Rep
	 */
	private interface CaseManipulator {

		/**
		 * Gets the selected text from the given document and changes it.
		 * 
		 * @param model the given document
		 */
		default void checkAllChars(DefaultMultipleDocumentModel model) {
			JTextArea area = model.getCurrentDocument().getTextComponent();
			Document doc = area.getDocument();
			int len = Math.abs(area.getCaret().getDot()-area.getCaret().getMark());
			int offset = 0;
			if(len!=0) {
				offset = Math.min(area.getCaret().getDot(),area.getCaret().getMark());
			} else {
				len = doc.getLength();
			}
			try {
				String text = doc.getText(offset, len);
				text = changeText(text);
				doc.remove(offset, len);
				doc.insertString(offset, text, null);
			} catch(BadLocationException ex) {
				ex.printStackTrace();
			}
		}

		/**
		 * Loops through all the characters from the given text 
		 * and calls the changeCharacter method.
		 * 
		 * @see changeCharacter
		 * @param text the given text
		 * @return String representation of the transformed text
		 */
		default String changeText(String text) {
			char[] znakovi = text.toCharArray();
			for(int i = 0; i < znakovi.length; i++) {
				char c = znakovi[i];
				znakovi[i] = changeCharacter(c);
			}
			return new String(znakovi);
		}

		/**
		 * Changes the given character with some transformation.
		 * 
		 * @param c the given character
		 * @return the transformed character
		 */
		public char changeCharacter(char c);
	}

	/**
	 * Inverts the case in the highlighted text.
	 */
	private Action invertAction = new LocalizableAction( "invert", flp ) {

		private static final long serialVersionUID = 1L;

		@Override
		public void actionPerformed(ActionEvent e) {

			new CaseManipulator() {

				@Override
				public char changeCharacter(char c) {
					if(Character.isLowerCase(c)) 
						return Character.toUpperCase(c);

					return Character.toLowerCase(c);
				}

			}.checkAllChars(m);

		}

	};

	/**
	 * Converts the selected text to upper case characters.
	 */
	private Action toUpperAction = new LocalizableAction( "toUpper", flp ) {

		private static final long serialVersionUID = 1L;

		@Override
		public void actionPerformed(ActionEvent e) {
			CaseManipulator cm = (c) -> Character.toUpperCase(c);
			cm.checkAllChars(m);
		}

	};


	/**
	 * Converts the selected text to lower case characters.
	 */
	private Action toLowerAction = new LocalizableAction( "toLower", flp ) {

		private static final long serialVersionUID = 1L;

		@Override
		public void actionPerformed(ActionEvent e) {

			CaseManipulator cm = (c) -> Character.toLowerCase(c);
			cm.checkAllChars(m);

		}

	};



	/**
	 * Interface used for strategy design patter.
	 * Reads the selected lines and changes them depending on the getChanged method.
	 * 
	 * @author Ivan Rep
	 */
	private interface TextChanger {

		/**
		 * Gets the selected text from the given models' text area
		 * and calls the getChanged method.
		 * 
		 * @see getChanged
		 * @param model the given document
		 */
		default void changeText(DefaultMultipleDocumentModel model) {

			JTextComponent c = model.getCurrentDocument().getTextComponent();
			Document doc = c.getDocument();
			Element root = doc.getDefaultRootElement();
			int rowStart, rowEnd;
			rowStart = root.getElementIndex(c.getSelectionStart());
			rowEnd = root.getElementIndex(c.getSelectionEnd());

			int offsetStart = 0, offsetEnd = 0;
			List<String> list = new ArrayList<>();

			for(int i=rowStart; i<rowEnd+1; i++) {

				Element e = root.getElement(i);

				if( i==rowStart ) offsetStart = e.getStartOffset();
				else if ( i==rowEnd ) offsetEnd = e.getEndOffset();

				try {
					list.add( doc.getText(e.getStartOffset(), e.getEndOffset()-e.getStartOffset() ) );
				} catch (BadLocationException e1) {	}
			}

			String content = getChangedText(list);

			try {
				doc.remove(offsetStart,offsetEnd-offsetStart-1);
				doc.insertString(offsetStart, content, null);
			} catch (BadLocationException e) {
				e.printStackTrace();
			}

		}

		/**
		 * Transforms the given list of lines.
		 * 
		 * @param list the given list of lines
		 * @return a single String with the transformed and concatenated lines
		 */
		public String getChangedText(List<String> list);

	}

	/**
	 * Action used for sorting the selected lines in ascending order.
	 */
	private Action sortAscending = new LocalizableAction( "sortAscending", flp ) {

		private static final long serialVersionUID = 1L;

		@Override
		public void actionPerformed(ActionEvent e) {
			TextChanger s = new TextChanger() {
				@Override
				public String getChangedText(List<String> list) {
					String currLang = flp.getCurrentLanguage();
					Locale locale = new Locale(currLang);
					Collator langCollator = Collator.getInstance(locale);

					Comparator<String> comp = (w1, w2) -> langCollator.compare(w1, w2);
					List<String> sorted = list.stream().sorted(comp).collect(Collectors.toList());

					return sorted.stream().collect(Collectors.joining());
				}
			};
			s.changeText(m);
		}

	};

	/**
	 * Action used for sorting the selected lines in descending order.
	 */
	private Action sortDescending = new LocalizableAction( "sortDescending", flp ) {

		private static final long serialVersionUID = 1L;

		@Override
		public void actionPerformed(ActionEvent e) {

			TextChanger s = new TextChanger() {
				@Override
				public String getChangedText(List<String> list) {
					String currLang = flp.getCurrentLanguage();
					Locale locale = new Locale(currLang);
					Collator langCollator = Collator.getInstance(locale);

					Comparator<String> comp = (w1, w2) -> langCollator.compare(w2, w1);
					List<String> sorted = list.stream().sorted(comp).collect(Collectors.toList());

					return sorted.stream().collect(Collectors.joining());
				}
			};
			s.changeText(m);

		}

	};

	/**
	 * Action used for removing repeating lines.
	 */
	private Action unique = new LocalizableAction( "unique", flp ) {

		private static final long serialVersionUID = 1L;

		@Override
		public void actionPerformed(ActionEvent e) {

			TextChanger s = new TextChanger() {
				@Override
				public String getChangedText(List<String> list) {

					LinkedHashSet<String> set = new LinkedHashSet<>(list);

					return set.stream().collect(Collectors.joining());
				}
			};
			s.changeText(m);

		}

	};

	/**
	 * Creates the actions' accelerator and mnemonic keys.
	 */
	private void createActions() {
		createBlankDocumentAction.putValue(Action.ACCELERATOR_KEY, KeyStroke.getKeyStroke("control 1")); 
		createBlankDocumentAction.putValue(Action.MNEMONIC_KEY, KeyEvent.VK_1); 

		saveAsDocumentAction.putValue(Action.ACCELERATOR_KEY, KeyStroke.getKeyStroke("control 2")); 
		saveAsDocumentAction.putValue(Action.MNEMONIC_KEY, KeyEvent.VK_2); 

		closeDocumentAction.putValue(Action.ACCELERATOR_KEY, KeyStroke.getKeyStroke("control 3")); 
		closeDocumentAction.putValue(Action.MNEMONIC_KEY, KeyEvent.VK_3); 

		cutAction.putValue(Action.ACCELERATOR_KEY, KeyStroke.getKeyStroke("control X")); 
		cutAction.putValue(Action.MNEMONIC_KEY, KeyEvent.VK_X); 

		copyAction.putValue(Action.ACCELERATOR_KEY, KeyStroke.getKeyStroke("control C")); 
		copyAction.putValue(Action.MNEMONIC_KEY, KeyEvent.VK_C); 

		pasteAction.putValue(Action.ACCELERATOR_KEY, KeyStroke.getKeyStroke("control V")); 
		pasteAction.putValue(Action.MNEMONIC_KEY, KeyEvent.VK_V); 

		infoAction.putValue(Action.ACCELERATOR_KEY, KeyStroke.getKeyStroke("control I")); 
		infoAction.putValue(Action.MNEMONIC_KEY, KeyEvent.VK_I); 

		openDocumentAction.putValue( Action.ACCELERATOR_KEY, KeyStroke.getKeyStroke("control O")); 
		openDocumentAction.putValue( Action.MNEMONIC_KEY, KeyEvent.VK_O); 

		saveDocumentAction.putValue( Action.ACCELERATOR_KEY, KeyStroke.getKeyStroke("control S")); 
		saveDocumentAction.putValue( Action.MNEMONIC_KEY, KeyEvent.VK_S); 

		invertAction.putValue( Action.ACCELERATOR_KEY, KeyStroke.getKeyStroke("control 4")); 
		invertAction.putValue( Action.MNEMONIC_KEY, KeyEvent.VK_4);

		toUpperAction.putValue( Action.ACCELERATOR_KEY, KeyStroke.getKeyStroke("control 5")); 
		toUpperAction.putValue( Action.MNEMONIC_KEY, KeyEvent.VK_5);

		toLowerAction.putValue( Action.ACCELERATOR_KEY, KeyStroke.getKeyStroke("control 6")); 
		toLowerAction.putValue( Action.MNEMONIC_KEY, KeyEvent.VK_6); 

		sortAscending.putValue( Action.ACCELERATOR_KEY, KeyStroke.getKeyStroke("control 7")); 
		sortAscending.putValue( Action.MNEMONIC_KEY, KeyEvent.VK_7); 

		sortDescending.putValue( Action.ACCELERATOR_KEY, KeyStroke.getKeyStroke("control 8")); 
		sortDescending.putValue( Action.MNEMONIC_KEY, KeyEvent.VK_8);

		unique.putValue( Action.ACCELERATOR_KEY, KeyStroke.getKeyStroke("control 9")); 
		unique.putValue( Action.MNEMONIC_KEY, KeyEvent.VK_9);

		exitAction.putValue(Action.ACCELERATOR_KEY, KeyStroke.getKeyStroke("control E")); 
		exitAction.putValue(Action.MNEMONIC_KEY, KeyEvent.VK_E); 
	}


	/**
	 * Creates the menus in the toolbars and the toolbars themselves.
	 */
	private void createMenus() {
		JMenuBar menuBar = new JMenuBar();

		JMenu fileMenu = new LJMenu("file", flp);
		menuBar.add(fileMenu);

		fileMenu.add(new JMenuItem(createBlankDocumentAction));
		fileMenu.add(new JMenuItem(openDocumentAction));
		fileMenu.add(new JMenuItem(saveDocumentAction));
		fileMenu.add(new JMenuItem(saveAsDocumentAction));
		fileMenu.add(new JMenuItem(closeDocumentAction));
		fileMenu.addSeparator();
		fileMenu.add(new JMenuItem(exitAction));

		JMenu editMenu = new LJMenu("edit", flp);
		menuBar.add(editMenu);

		editMenu.add(new JMenuItem(cutAction));
		editMenu.add(new JMenuItem(copyAction));
		editMenu.add(new JMenuItem(pasteAction));
		editMenu.add(new JMenuItem(infoAction));

		JMenu langMenu = new LJMenu("language", flp);
		menuBar.add(langMenu);

		JMenuItem i1 = createLangItem("hr");
		langMenu.add(i1);
		JMenuItem i2 = createLangItem("en");
		langMenu.add(i2);

		JMenu tools = new LJMenu("tools", flp);
		menuBar.add(tools);

		this.caseMenu = new LJMenu("case", flp);
		tools.add(caseMenu);
		this.caseMenu.setEnabled(false);
		caseMenu.add(new JMenuItem(toUpperAction));
		caseMenu.add(new JMenuItem(toLowerAction));
		caseMenu.add(new JMenuItem(invertAction));

		JMenu sortMenu = new LJMenu("sort", flp);
		tools.add(sortMenu);
		sortMenu.add(new JMenuItem(sortAscending));
		sortMenu.add(new JMenuItem(sortDescending));
		sortMenu.add(new JMenuItem(unique));

		this.setJMenuBar(menuBar);
	}

	/**
	 * Helper method that creates JMenuItem with the given language value.
	 * Sets the action listener.
	 * 
	 * @param lang the given language value
	 * @return a new JMenuItem object with the set language value
	 */
	private JMenuItem createLangItem(String lang) {
		JMenuItem i1 = new JMenuItem(lang);
		i1.addActionListener( (e) -> 
		LocalizationProvider.getInstance().setLanguage(lang)
				);
		return i1;
	}

	/**
	 * Creates the toolbars and connects them with the correct actionss.
	 */
	private void createToolbars() {
		JToolBar toolBar = new JToolBar();
		toolBar.setFloatable(true);

		toolBar.add(new JButton(createBlankDocumentAction));
		toolBar.add(new JButton(openDocumentAction));
		toolBar.add(new JButton(saveDocumentAction));
		toolBar.add(new JButton(saveAsDocumentAction));
		toolBar.add(new JButton(closeDocumentAction));
		toolBar.addSeparator();
		toolBar.add(new JButton(cutAction));
		toolBar.add(new JButton(copyAction));
		toolBar.add(new JButton(pasteAction));
		toolBar.add(new JButton(infoAction));
		toolBar.add(new JButton(exitAction));

		this.getContentPane().add(toolBar, BorderLayout.PAGE_START);
	}

	/**
	 * Main method
	 * 
	 * @param args not used here
	 */
	public static void main(String[] args) {
		SwingUtilities.invokeLater(new Runnable() {

			@Override
			public void run() {
				LocalizationProvider.getInstance().setLanguage("en");
				new JNotepadPP().setVisible(true);
			}
		});
	}
}