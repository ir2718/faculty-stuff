package hr.fer.oprpp1.hw08.jnotepadpp;

/**
 * Class represents a document listener.
 * 
 * @author Ivan Rep
 */
public interface SingleDocumentListener {

	/**
	 * Sets the icon of the given document's tab.
	 * 
	 * @param model the given document
	 */
	void documentModifyStatusUpdated(SingleDocumentModel model);
	
	/**
	 * Sets the title and tooltip for the given document's tab.
	 * 
	 * @param model the given document
	 */
	void documentFilePathUpdated(SingleDocumentModel model);

}