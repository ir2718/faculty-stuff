package hr.fer.oprpp1.hw08.jnotepadpp;

import java.nio.file.Path;

/**
 * Interface for representing a multiple document model.
 * 
 * @author Ivan Rep
 */
public interface MultipleDocumentModel  extends Iterable<SingleDocumentModel> {

	/**
	 * Creates and returns a new SingleDocumentModel object.
	 * 
	 * @return a new document
	 */
	SingleDocumentModel createNewDocument();
	
	/**
	 * Returns the current document.
	 * 
	 * @return the current document.
	 */
	SingleDocumentModel getCurrentDocument();
	
	/**
	 * Loads the document from the given path.
	 * Returns a new document with the given path
	 * and the content read from it.
	 * 
	 * @param path the given path
	 * @return new document with the given path and the content
	 */
	SingleDocumentModel loadDocument(Path path);
	
	/**
	 * Saves the given document model to the given path.
	 * 
	 * @param model the given document model
	 * @param newPath the given path
	 */
	void saveDocument(SingleDocumentModel model, Path newPath);
	
	/**
	 * Closes the given document model.
	 * 
	 * @param model the given document model
	 */
	void closeDocument(SingleDocumentModel model);
	
	/**
	 * Registers the given document listener.
	 * 
	 * @param l the given document listener
	 */
	void addMultipleDocumentListener(MultipleDocumentListener l);
	
	/**
	 * Unregisters the given document listener.
	 * 
	 * @param l the given document listener
	 */
	void removeMultipleDocumentListener(MultipleDocumentListener l);
	
	/**
	 * Returns the amount of documents in the current DefaultMultipleModel.
	 * 
	 * @return the amount of documents in the model
	 */
	int getNumberOfDocuments();
	
	/**
	 * Returns the document at the given index.
	 * 
	 * @param index the given index
	 * @return the document at the given index
	 */
	SingleDocumentModel getDocument(int index);

}