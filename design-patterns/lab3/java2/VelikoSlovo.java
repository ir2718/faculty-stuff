package MyTextEditor.plugins;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import MyTextEditor.ClipboardStack;
import MyTextEditor.Plugin;
import MyTextEditor.TextEditorModel;
import MyTextEditor.UndoManager;

public class VelikoSlovo implements Plugin {

	@Override
	public String getName() {
		return "Veliko slovo";
	}

	@Override
	public String getDescription() {
		return "U cijelom dokumentu mijenja svako prvo slovo rijeci u veliko slovo.";
	}

	@Override
	public void execute(TextEditorModel model, UndoManager undoManager, ClipboardStack clipboardStack) {
		List<String> lines = model.getLines();
		
		for(int i=0; i<lines.size(); i++) {
			String s = lines.remove(i);
			String[] words = s.split(" ");

			for(int j=0; j<words.length; j++) {
				if(!words[j].isEmpty() && Character.isAlphabetic(words[j].charAt(0)))
					words[j] = String.valueOf(words[j].charAt(0)).toUpperCase() + words[j].substring(1, words[j].length());
			}
			
			lines.add(i, Arrays.stream(words).collect(Collectors.joining(" ")));
		}
		
		model.notifyTextObservers();
	}

}
