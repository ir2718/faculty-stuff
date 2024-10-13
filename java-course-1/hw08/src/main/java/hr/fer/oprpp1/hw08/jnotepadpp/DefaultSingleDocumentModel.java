package hr.fer.oprpp1.hw08.jnotepadpp;

import java.nio.file.Path;
import java.util.ArrayList;
import javax.swing.JTextArea;
import javax.swing.event.DocumentEvent;
import javax.swing.event.DocumentListener;

/**
 * Class represents a model for exactly one document.
 * 
 * @author Ivan Rep
 */
public class DefaultSingleDocumentModel implements SingleDocumentModel {
	
	private Path path;
	private ArrayList<SingleDocumentListener> listeners;
	private JTextArea area;
	private boolean modified;

	/**
	 * Constructor.
	 * Sets the path and content.
	 * 
	 * @param path the given path
	 * @param content the given content
	 */
	public DefaultSingleDocumentModel(Path path, String content) {
		this.path = path;
		this.listeners = new ArrayList<>();
		this.area = new JTextArea(content);
		this.area.getDocument().addDocumentListener(new DocumentListener() {

			@Override
			public void insertUpdate(DocumentEvent e) {
				modified = true;
				notifyModified();
			}

			@Override
			public void removeUpdate(DocumentEvent e) {	insertUpdate(e); }

			@Override
			public void changedUpdate(DocumentEvent e) { insertUpdate(e); }
			
		});
	}

	/**
	 * Returns the document's text component.
	 * 
	 * @return the document's text component
	 */
	@Override
	public JTextArea getTextComponent() {
		return this.area;
	}

	/**
	 * Returns the document's file path.
	 * 
	 * @return the document's file path
	 */
	@Override
	public Path getFilePath() {
		return this.path;
	}

	/**
	 * Sets the file path to the given path value.
	 * 
	 * @param path the given path value
	 */
	@Override
	public void setFilePath(Path path) {
		if( path == null ) throw new IllegalArgumentException("Given path can't be null.");
		this.path = path;
		this.notifyPath();
	}

	/**
	 * Checks if the document is modified.
	 * 
	 * @return true if modified, otherwise false
	 */
	@Override
	public boolean isModified() {
		return this.modified;
	}

	/**
	 * Sets the modified flag to the given boolean value.
	 * 
	 * @param path the given boolean value
	 */
	@Override
	public void setModified(boolean modified) {
		this.modified = modified;
		this.notifyModified();
	}

	/**
	 * Registers the given document listener.
	 * 
	 * @param l the given document listener
	 */
	@Override
	public void addSingleDocumentListener(SingleDocumentListener l) {
		listeners.add( l );
	}

	/**
	 * Unregisters the given document listener.
	 * 
	 * @param l the given document listener
	 */
	@Override
	public void removeSingleDocumentListener(SingleDocumentListener l) {
		listeners.remove( l );
	}
	
	/**
	 * Notifies listeners that the path has changed.
	 */
	private void notifyPath() {
		listeners.forEach( l -> l.documentFilePathUpdated(this) );
	}
	
	/**
	 * Notifies listeners that the modified flag has changed.
	 */
	private void notifyModified() {
		listeners.forEach( l -> l.documentModifyStatusUpdated(this) );
	}
	
}
