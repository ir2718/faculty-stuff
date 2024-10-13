package hr.fer.oprpp1.hw08.jnotepadpp;

import java.nio.file.Path;

import javax.swing.JTextArea;

/**
 * Interface for reresenting exactly one document.
 * 
 * @author Ivan Rep
 */
public interface SingleDocumentModel {

	/**
	 * Returns the document's text component.
	 * 
	 * @return the document's text component
	 */
	JTextArea getTextComponent();
	
	/**
	 * Returns the document's file path.
	 * 
	 * @return the document's file path
	 */
	Path getFilePath();
	
	/**
	 * Sets the file path to the given path value.
	 * 
	 * @param path the given path value
	 */
	void setFilePath(Path path);
	
	/**
	 * Checks if the document is modified.
	 * 
	 * @return true if modified, otherwise false
	 */
	boolean isModified();
	
	/**
	 * Sets the modified flag to the given boolean value.
	 * 
	 * @param path the given boolean value
	 */
	void setModified(boolean modified);
	
	/**
	 * Registers the given document listener.
	 * 
	 * @param l the given document listener
	 */
	void addSingleDocumentListener(SingleDocumentListener l);
	
	/**
	 * Unregisters the given document listener.
	 * 
	 * @param l the given document listener
	 */
	void removeSingleDocumentListener(SingleDocumentListener l);

}
