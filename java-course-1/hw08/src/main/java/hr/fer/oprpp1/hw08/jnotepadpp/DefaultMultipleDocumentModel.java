package hr.fer.oprpp1.hw08.jnotepadpp;

import java.awt.Graphics;
import java.awt.Image;
import java.awt.event.ComponentEvent;
import java.awt.event.ComponentListener;
import java.awt.image.ImageObserver;
import java.awt.image.ImageProducer;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.stream.Collectors;

import javax.swing.ImageIcon;
import javax.swing.JButton;
import javax.swing.JScrollPane;
import javax.swing.JTabbedPane;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import javax.swing.tree.DefaultMutableTreeNode;

/**
 * Class represents a model for holding one or more SingleDocumentModels.
 * 
 * @author Ivan Rep
 */
public class DefaultMultipleDocumentModel extends JTabbedPane implements MultipleDocumentModel {

	private ArrayList<SingleDocumentModel> documents;
	private SingleDocumentModel currDoc;
	private ArrayList<MultipleDocumentListener> multipleListeners;

	private ImageIcon GREEN_DISKETTE;
	private ImageIcon RED_DISKETTE;

	/**
	 * Constructor. Initializes listeners and lists.
	 */
	public DefaultMultipleDocumentModel() {
		documents = new ArrayList<>();
		currDoc = null;
		multipleListeners = new ArrayList<>();

		this.addChangeListener(new ChangeListener() {
			@Override
			public void stateChanged(ChangeEvent e) {
				SingleDocumentModel previous = currDoc;
				DefaultMultipleDocumentModel m = DefaultMultipleDocumentModel.this;
				if( m.getSelectedIndex() == -1 ) {
					currDoc = null;
				} else {
					currDoc = documents.get( m.getSelectedIndex() );
					DefaultMultipleDocumentModel.this.setToolTipTextAt( m.getSelectedIndex(), 
							currDoc.getFilePath() == null ? "(unnamed)" : currDoc.getFilePath().toString() );
				}
				notifyChangedCurrentDocument(previous, currDoc);
			}
		});

		Image img = this.loadIcon("icons/green.png")
				.getImage()
				.getScaledInstance(10, 10, java.awt.Image.SCALE_SMOOTH);
		GREEN_DISKETTE = new ImageIcon(img);

		Image img2 = this.loadIcon("icons/red.png")
				.getImage()
				.getScaledInstance(10, 10, java.awt.Image.SCALE_SMOOTH);
		RED_DISKETTE = new ImageIcon(img2);

	}

	/**
	 * Returns an iterator for all the documents.
	 * 
	 * @return an iterator for all the documents
	 */
	@Override
	public Iterator<SingleDocumentModel> iterator() {
		return this.documents.iterator();
	}

	/**
	 * Creates and returns a new SingleDocumentModel object.
	 * 
	 * @return a new document
	 */
	@Override
	public SingleDocumentModel createNewDocument() {
		SingleDocumentModel newDoc = addDocument(null, "");
		return newDoc;
	}

	/**
	 * Adds a document with the given path and content and registers listeners.
	 * 
	 * @param path the given path
	 * @param s the content
	 * @return a new SingleDocumentModel object with the given values
	 */
	private SingleDocumentModel addDocument(Path path, String s) {
		currDoc = new DefaultSingleDocumentModel(path, s);

		currDoc.addSingleDocumentListener(new SingleDocumentListener() {

			@Override
			public void documentModifyStatusUpdated(SingleDocumentModel model) {
				DefaultMultipleDocumentModel m = DefaultMultipleDocumentModel.this;
				m.setIconAt(m.getSelectedIndex(),
						model.isModified() ? m.RED_DISKETTE : m.GREEN_DISKETTE );
			}

			@Override
			public void documentFilePathUpdated(SingleDocumentModel model) {
				DefaultMultipleDocumentModel m = DefaultMultipleDocumentModel.this;
				m.setToolTipTextAt(m.getSelectedIndex(), model.getFilePath().toString());
				m.setTitleAt(m.getSelectedIndex(), model.getFilePath().toString());
			}

		});

		this.documents.add(currDoc);
		this.notifyAdded(currDoc);

		String docName = path == null ? "(unnamed)" : path.toAbsolutePath().toString();
		JScrollPane pane = new JScrollPane(currDoc.getTextComponent());
		this.addTab(docName, GREEN_DISKETTE, pane, docName);
		this.setSelectedComponent( pane );

		return currDoc;
	}

	/**
	 * Returns the current document.
	 * 
	 * @return the current document.
	 */
	@Override
	public SingleDocumentModel getCurrentDocument() {
		return this.currDoc;
	}

	/**
	 * Loads the document from the given path.
	 * Returns a new document with the given path
	 * and the content read from it.
	 * 
	 * @param path the given path
	 * @return new document with the given path and the content
	 */
	@Override
	public SingleDocumentModel loadDocument(Path path) {
		if( path == null ) throw new IllegalArgumentException("Path can't be null.");

		if( this.documents!=null ) {

			List<SingleDocumentModel> models = this.documents
					.stream()
					.filter( d -> d.getFilePath()!=null && d.getFilePath().equals(path) )
					.collect(Collectors.toList());

			if(!models.isEmpty())  {
				this.notifyChangedCurrentDocument(currDoc, models.get(0));
				currDoc = models.get(0);
				this.setSelectedComponent(currDoc.getTextComponent().getParent());
				return currDoc;
			}

		}

		StringBuilder sb = new StringBuilder();

		readDocument(path, sb);

		addDocument(path, sb.toString());

		return currDoc;
	}

	/**
	 * Saves the given document model to the given path.
	 * 
	 * @param model the given document model
	 * @param newPath the given path
	 */
	@Override
	public void saveDocument(SingleDocumentModel model, Path newPath) {
		if(newPath == null) newPath = model.getFilePath();

		SingleDocumentModel temp = currDoc;

		currDoc.setFilePath(newPath);

		String text = model.getTextComponent().getText();

		try (OutputStream os = Files.newOutputStream(newPath)) {
			os.write(text.getBytes(StandardCharsets.UTF_8));
			os.close();
		} catch(IOException ex) {
			throw new IllegalStateException("Something went wrong while saving.");
		}

		model.setModified(false);

		notifyChangedCurrentDocument(temp, this.getCurrentDocument());

	}

	/**
	 * Closes the given document model.
	 * 
	 * @param model the given document model
	 */
	@Override
	public void closeDocument(SingleDocumentModel model) {
		if( this.documents.isEmpty() ) 
			return;

		this.notifyRemoved(this.currDoc);

		this.removeTabAt( this.documents.indexOf(model) );
		this.documents.remove( model );
		int index = this.documents.size()-1;
		this.currDoc = this.documents.isEmpty() ? null : this.documents.get(index);

	}

	/**
	 * Registers the given document listener.
	 * 
	 * @param l the given document listener
	 */
	@Override
	public void addMultipleDocumentListener(MultipleDocumentListener l) {
		this.multipleListeners.add(l);
	}

	/**
	 * Unregisters the given document listener.
	 * 
	 * @param l the given document listener
	 */
	@Override
	public void removeMultipleDocumentListener(MultipleDocumentListener l) {
		this.multipleListeners.remove(l);
	}

	/**
	 * Returns the amount of documents in the current DefaultMultipleModel.
	 * 
	 * @retrun the amount of documents in the model
	 */
	@Override
	public int getNumberOfDocuments() {
		return this.documents.size();
	}

	/**
	 * Returns the document at the given index.
	 * 
	 * @param index the given index
	 * @return the document at the given index
	 */
	@Override
	public SingleDocumentModel getDocument(int index) {
		return this.documents.get(index);
	}

	/**
	 * Helper method used for reading the document.
	 * 
	 * @param path path for the file
	 * @param sb StringBuilder object for saving the content
	 */
	private static void readDocument(Path path, StringBuilder sb) {

		try (InputStream is = Files.newInputStream(path)) {
			sb.append( new String(is.readAllBytes(), StandardCharsets.UTF_8) );
		} catch(IOException ex) { }

	}

	/**
	 * Helper method for loading icons from the given path.
	 * 
	 * @param s the given path
	 * @return a new ImageIcon object
	 */
	private ImageIcon loadIcon(String s) {

		InputStream is;
		byte[] bytes = null;
		
		try {
			is = this.getClass().getResourceAsStream(s);
			if(is==null) throw new IllegalArgumentException();
			bytes = is.readAllBytes();
			is.close();
		} catch( IOException | IllegalArgumentException exc ) { 
			throw new IllegalArgumentException("The given path doesn't exist or is null.");
		}
		
		return new ImageIcon(bytes);
	}

	/**
	 * Notifies the listeners that the current document changed.
	 * 
	 * @param previous the previous document
	 * @param current the current document
	 */
	private void notifyChangedCurrentDocument(  SingleDocumentModel previous, SingleDocumentModel current ) {
		this.multipleListeners.forEach( l -> l.currentDocumentChanged(previous, current) );
	}

	/**
	 * Notifies the listeners that a document was added.
	 * 
	 * @param model the added document
	 */
	private void notifyAdded( SingleDocumentModel model ) {
		this.multipleListeners.forEach( l -> l.documentAdded(model) );
	}

	/**
	 * Notifies the listeners that a document was removed.
	 * 
	 * @param model the removed document
	 */
	private void notifyRemoved( SingleDocumentModel model ) {
		this.multipleListeners.forEach( l -> l.documentRemoved(model) );
	}

}
