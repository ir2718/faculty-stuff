package MyTextEditor.plugins;

import java.util.List;

import javax.swing.JDialog;
import javax.swing.JOptionPane;

import MyTextEditor.ClipboardStack;
import MyTextEditor.Plugin;
import MyTextEditor.TextEditorModel;
import MyTextEditor.UndoManager;

public class Statistika extends JDialog implements Plugin {
	
	public Statistika() {
		setDefaultCloseOperation(DISPOSE_ON_CLOSE);
		setModal(true);
		pack();
	}

	@Override
	public String getName() {
		return "Statistika";
	}

	@Override
	public String getDescription() {
		return "Plugin broji koliko ima redaka, rijeci i slova u dokumentu i to prikazuje korisniku u dijalogu.";
	}

	@Override
	public void execute(TextEditorModel model, UndoManager undoManager, ClipboardStack clipboardStack) {
		List<String> l = model.getLines();
		int rows = l.size();
		int wordCount = 0;
		int characterCount = 0;
		
		for(String s : l) {
			String[] words = s.trim().split("\\s+");
			wordCount += words.length;
		
			for(String w : words) 
				characterCount += w.length();
		}

		JOptionPane.showMessageDialog(this.getParent(), "Character count: "+ characterCount +"\nWord count: "+ wordCount +"\nRow count: "+ rows);
	}
}
