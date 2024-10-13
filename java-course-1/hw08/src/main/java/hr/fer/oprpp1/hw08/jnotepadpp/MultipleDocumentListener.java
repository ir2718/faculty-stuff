package hr.fer.oprpp1.hw08.jnotepadpp;

/**
 * Class represents a listener for the multiple document model.
 * 
 * @author Ivan Rep
 */
public interface MultipleDocumentListener {
	
	/**
	 * Sets the window title and the tab name.
	 * 
	 * @param previousModel the previous model
	 * @param currentModel the current model
	 */
	void currentDocumentChanged(SingleDocumentModel previousModel, SingleDocumentModel currentModel);
	
	/**
	 * Adds a caret listener to the given document model.
	 * 
	 * @param model the given document model
	 */
	void documentAdded(SingleDocumentModel model);
	
	/**
	 * Removes the caret listener from the given document model..
	 * 
	 * @param model the given document model
	 */
	void documentRemoved(SingleDocumentModel model);
	
}