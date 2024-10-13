package MyTextEditor;

import java.util.Arrays;
import static java.lang.Math.*;

import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;

public class TextEditorModel {

	private List<String> lines;
	private LocationRange selectionRange;
	private Location cursorLocation;
	private ClipboardStack clipboard;

	private List<CursorObserver> cursorObservers;
	private List<TextObserver> textObservers;
	private List<SelectionObserver> selectionObservers;

	public TextEditorModel(String text) {
		this.selectionObservers = new LinkedList<>();
		this.cursorObservers = new LinkedList<>();
		this.textObservers = new LinkedList<>();
		this.lines = new LinkedList<>(Arrays.asList(text.split("\n")));
		this.clipboard = new ClipboardStack();
	}

	public void setCursorLocation(Location cursorLocation) {
		this.cursorLocation = cursorLocation;
	}

	public List<String> getLines() {
		return lines;
	}

	public Location getCursorLocation() {
		return cursorLocation;
	}

	public ClipboardStack getClipboardStack() {
		return this.clipboard;
	}

	public Iterator<String> allLines() {
		return new MyIterator<String>(this.lines);
	}

	public Iterator<String> linesRange(int index1, int index2)  {
		return new MyIterator<String>(this.lines.subList(index1, index2));
	}

	public void addCursorObserver(CursorObserver o) {
		this.cursorObservers.add(o);
	}

	public void removeCursorObserver(CursorObserver o) {
		this.cursorObservers.remove(o);
	}

	public void notifyCursorObservers() {
		this.cursorObservers.forEach(o -> o.updateCursorLocation(this.cursorLocation));
	}

	public void addTextObserver(TextObserver o) {
		this.textObservers.add(o);
	}

	public void removeTextObserver(TextObserver o) {
		this.textObservers.remove(o);
	}

	public void notifyTextObservers() {
		this.textObservers.forEach(o -> o.updateText());
	}

	public void notifySelectionObservers() {
		this.selectionObservers.forEach(o -> o.update());
	}

	public void addSelectionObserver(SelectionObserver o) {
		this.selectionObservers.add(o);
	}

	public void removeSelectionObserver(SelectionObserver o) {
		this.selectionObservers.remove(o);
	}

	public void notifyAllObservers() {
		this.notifyCursorObservers();
		this.notifyTextObservers();
		this.notifySelectionObservers();
	}

	public void moveLeft() {
		int x = this.cursorLocation.getX();
		int y = this.cursorLocation.getY();
		if(x == 0 && y == 0) return;		

		if(x == 0) {
			this.cursorLocation.setY(y-1);
			this.cursorLocation.setX(lines.get(y-1).length());
		} else {
			this.cursorLocation.setX(x-1);
		}

		this.notifyCursorObservers();
		this.notifySelectionObservers();
	}

	public void moveRight() {
		if(lines.isEmpty()) return;

		int x = this.cursorLocation.getX();
		int y = this.cursorLocation.getY();
		String currStr = lines.get(y);
		if(x == currStr.length() && y == lines.size()-1) return;

		if(x == currStr.length()) {
			this.cursorLocation.setY(y+1);
			this.cursorLocation.setX(0);
		} else {
			this.cursorLocation.setX(x+1);
		}

		this.notifyCursorObservers();
		this.notifySelectionObservers();

	}

	public void moveUp() {
		int y = this.cursorLocation.getY();
		if (y == 0) return;		

		this.cursorLocation.setY(y-1);
		this.cursorLocation.setX(min(lines.get(y-1).length(), this.cursorLocation.getX()));
		this.notifyCursorObservers();
		this.notifySelectionObservers();
	}

	public void moveDown() {
		int y = this.cursorLocation.getY();
		if(y == lines.size() - 1) return;	

		this.cursorLocation.setY(y+1);
		this.cursorLocation.setX(min(lines.get(y+1).length(), this.cursorLocation.getX()));
		this.notifyCursorObservers();
		this.notifySelectionObservers();
	}

	public void moveCursorResetSelection(Runnable runnable) {
		runnable.run();
		this.setSelectionRange(null);
	}

	public void deleteBefore() {
		if(this.selectionRange == null) {
			int x = this.cursorLocation.getX();
			int y = this.cursorLocation.getY();
			if (x == 0 && y == 0 || lines.isEmpty()) 
				return;

			String line = lines.remove(y);
			if(x == 1 && y == 0 && line.length() == 1) {
				this.cursorLocation.setX(0);
			} else if (x == 0 && line.length() > 0) {
				String lineBefore = this.lines.remove(y-1);
				this.lines.add(y-1, lineBefore + line);
				this.cursorLocation.setY(y-1);
				this.cursorLocation.setX(lineBefore.length());
			} else if (line.length() >= 1) {
				String newLine = line.substring(0, x<=0 ? 0 : x-1) + line.substring(x<=0 ? 0 : x);
				lines.add(y, newLine);
				this.cursorLocation.setX(x<=0 ? 0 : x-1);
			} else {
				this.cursorLocation.setY(max(0, y-1));
				this.cursorLocation.setX(this.lines.get(y-1).length());
			}
		} else {
			Location start = this.selectionRange.sort().getStart();
			deleteRange(this.selectionRange);
			this.setCursorLocation(new Location(start.getX(), start.getY()));
		}

		this.notifyAllObservers();
	}

	public void deleteAfter() {
		if(this.selectionRange == null) {
			int x = this.cursorLocation.getX();
			int y = this.cursorLocation.getY();
			if (y == lines.size()-1 && x == lines.get(lines.size()-1).length()) 
				return;

			String line = lines.remove(y);
			if (y < lines.size() && x == line.length()) {
				String lineAfter = this.lines.remove(y);
				this.lines.add(y, line + lineAfter);
			} else {
				String newLine = line.substring(0, x) + line.substring(x+1);
				lines.add(y, newLine);
			}
		} else {
			deleteRange(this.selectionRange);
		}
		this.notifyTextObservers();
		this.notifySelectionObservers();
	}

	public void deleteRange(LocationRange r) {
		r = r.sort();
		int x1 = r.getStart().getX();
		int y1 = r.getStart().getY();
		int x2 = r.getEnd().getX();
		int y2 = r.getEnd().getY();

		if(y1 == y2) {
			String l = this.lines.remove(y1);
			l = l.substring(0, x1) + l.substring(x2);
			this.lines.add(y1, l);
		} else {
			String l1 = this.lines.get(y1).substring(0, x1);
			String l2 = this.lines.get(y2).substring(x2);

			for(int i=0, j=0; j<=y2-y1 ;j++)
				this.lines.remove(i+y1);

			String l = l1 + l2;
			this.lines.add(y1, l);
		}
	}

	public LocationRange getSelectionRange() {
		return this.selectionRange;
	}

	public void setSelectionRange(LocationRange range) {
		this.selectionRange = range;
	}

	public void moveCursorShift(Runnable runnable) {
		LocationRange r = this.getSelectionRange();

		if (r == null) {
			Location lStart = new Location(this.getCursorLocation());
			runnable.run();
			Location lEnd = new Location(this.getCursorLocation());
			this.setSelectionRange(new LocationRange(lStart, lEnd));
		} else {
			runnable.run();
			Location end = new Location(this.getCursorLocation());
			this.selectionRange.setEnd(end);
		}

		this.notifyAllObservers();
	}
	public void insert(char c) {
		if(this.selectionRange != null) {
			this.deleteBefore();
			EditAction edit = new InsertCharEditAction(c, this.cursorLocation);
			UndoManager.getInstance().push(edit);
			edit.executeDo();
			this.setSelectionRange(null);
			this.notifyAllObservers();
			return;
		}

		EditAction edit = new InsertCharEditAction(c, this.cursorLocation);
		UndoManager.getInstance().push(edit);
		edit.executeDo();

		this.notifyAllObservers();
	}

	public void insert(String text) {
		if(this.selectionRange != null) {
			this.deleteBefore();
			EditAction edit = new InsertStringEditAction(text, this.lines, this.cursorLocation);
			UndoManager.getInstance().push(edit);
			edit.executeDo();
			this.setSelectionRange(null);
			this.notifyAllObservers();
			return;
		}
		
		EditAction edit = new InsertStringEditAction(text, this.lines, this.cursorLocation);
		edit.executeDo();
		UndoManager.getInstance().push(edit);
		
		this.notifyAllObservers();
	}


	public void copy() {
		LocationRange r = this.selectionRange;
		if(r == null) return;

		r = r.sort();
		int y1 = r.getStart().getY();
		int y2 = r.getEnd().getY();
		int x1 = r.getStart().getX();
		int x2 = r.getEnd().getX();

		if(y1 == y2) {
			this.clipboard.push(this.lines.get(y1).substring(x1, x2));
			return;
		}

		String copyString = "";

		String first = this.lines.get(y1).substring(x1);
		copyString += first + "\n";

		for (int i=y1 + 1; i<y2 - 1; i++)
			copyString += this.lines.get(i) + "\n";

		String last = this.lines.get(y2).substring(0, x2);
		copyString += last;

		this.clipboard.push(copyString);
	}

	public void paste() {
		if(this.clipboard.isEmpty()) return;
		this.insert(this.clipboard.peek());
	}

	public void crop() {
		LocationRange r = this.selectionRange;
		if(r == null) return;

		r = r.sort();
		int y1 = r.getStart().getY();
		int y2 = r.getEnd().getY();
		int x1 = r.getStart().getX();
		int x2 = r.getEnd().getX();

		List<String> listOld = new LinkedList<>(lines);
		Location clOld = new Location(cursorLocation);
		
		if(y1 == y2) {
			String line = this.lines.remove(y1);
			this.clipboard.push(line.substring(x1, x2));
			this.lines.add(y1, line.substring(0, x1) + line.substring(x2));
			this.cursorLocation.setX(x1);
			this.selectionRange = null;

			this.notifyAllObservers();
			return;
		}

		String cropString = "";

		String line = this.lines.remove(y1);
		cropString += line.substring(x1) + "\n";

		for(int j=1; j<y2 - y1; j++)
			cropString += this.lines.remove(y1) + "\n";

		String last = this.lines.remove(y1);
		cropString += last.substring(0, x2);
		this.clipboard.push(cropString);

		String newLine = line.substring(0, x1) + last.substring(x2);
		this.lines.add(y1, newLine);

		this.cursorLocation.setX(x1);
		this.cursorLocation.setY(y1);
		this.selectionRange = null;

		EditActionClass edit = new EditActionClass(listOld, this.lines, clOld, this.cursorLocation);
		UndoManager.getInstance().push(edit);
		
		this.notifyAllObservers();
	}

	public void pasteCtrl() {
		List<String> listOld = new LinkedList<>(lines);
		Location clOld = new Location(cursorLocation);
		
		if(this.clipboard.isEmpty()) return;
		this.insert(this.clipboard.pop());
		
		EditActionClass edit = new EditActionClass(listOld, this.lines, clOld, this.cursorLocation);
		UndoManager.getInstance().push(edit);
	}

	public void cursorToDocumentStart() {
		this.cursorLocation.setToStart();
		this.notifyAllObservers();
	}

	public void cursorToDocumentEnd() {
		if (this.lines.isEmpty()) return;

		int last = this.getLines().size() - 1;
		this.setCursorLocation(new Location(this.lines.get(last).length(), last));
		this.notifyAllObservers();
	}

	public void clearDocument() {
		this.lines.clear();
		this.cursorLocation.setToStart();
		this.notifyAllObservers();
	}

	public void deleteSelected() {
		if(this.selectionRange == null) return;

		Location start = new Location(this.selectionRange.sort().getStart());
		this.deleteRange(this.selectionRange);
		this.selectionRange = null;
		this.setCursorLocation(start);
		this.notifyAllObservers();
	}

	public void readFromPath(Path filePath) throws IOException {
		this.lines = Files.readAllLines(filePath);

		this.cursorLocation.setToStart();
		if (this.selectionRange != null) {
			this.selectionRange = null;
		}

		this.notifyAllObservers();
	}

	public void saveLines(Path filePath) throws IOException {
		Files.write(filePath, this.lines, Charset.defaultCharset());
	}

	public void undo() {
		this.setSelectionRange(null);
		UndoManager.getInstance().undo();
		notifyAllObservers();
	}

	public void redo() {
		this.setSelectionRange(null);
		UndoManager.getInstance().redo();
		notifyAllObservers();
	}

	private class EditActionClass implements EditAction {

		private List<String> listOld;
		private List<String> listNew;
		private Location clOld;
		private Location clNew;
		
		public EditActionClass(List<String> listOld, List<String> listNew, Location clOld, Location clNew) {
			this.listOld = new LinkedList<>(listOld);
			this.listNew = new LinkedList<>(listNew);
			this.clOld = new Location(clOld);
			this.clNew = new Location(clNew);
		}

		@Override
		public void executeDo() {
			lines = listNew;
			cursorLocation = clNew;
		}

		@Override
		public void executeUndo() {
			lines = listOld;
			cursorLocation = clOld;
		}
	}
	
	private class InsertCharEditAction implements EditAction {
		
		private char c;
		private Location cl;
		
		public InsertCharEditAction(char c, Location cl) {
			this.c = c;
			this.cl = cl;
		}
		
		@Override
		public void executeDo() {
			String curr = lines.isEmpty() ? "" : lines.remove(cl.getY());

			if (c != '\n') {
				curr = curr.substring(0, cl.getX()) + String.valueOf(c) + curr.substring(cl.getX());
				lines.add(cl.getY(), curr);
				setCursorLocation(new Location(cl.getX() + 1, cl.getY()));
			} else {
				String curr2 = curr.substring(cl.getX());
				curr = curr.substring(0, cl.getX());
				lines.add(cl.getY(), curr);
				lines.add(cl.getY() + 1, curr2);
				setCursorLocation(new Location(0, cl.getY() + 1));
			}
		}

		@Override
		public void executeUndo() {
			if(c == '\n') {
				int y = cl.getY();
				int x = lines.get(y).length();
				String before = lines.get(y);
				String curr = lines.remove(y + 1);
				lines.set(y, before + curr);
				cursorLocation.setY(y);
				cursorLocation.setX(x);
			} else {
				String s = lines.get(cl.getY());
				lines.set(cl.getY(), s.substring(0, cl.getX()) + s.substring(cl.getX() + 1));
				if(cl.getX() == 0) {
					int beforeLen = lines.get(cl.getX()).length();
					cursorLocation.setX(beforeLen);
					cursorLocation.setY(cl.getY() - 1);
				} 
				else {
					cursorLocation.setX(cl.getX());
					cursorLocation.setY(cl.getY());
				}
			}
		}
		
	}
	
	
	private class InsertStringEditAction implements EditAction {

		private List<String> listOld;
		private Location clOld;
		private String text;
		
		public InsertStringEditAction(String text, List<String> listOld, Location clOld) {
			this.listOld = new LinkedList<>(listOld);
			this.clOld = new Location(clOld);
			this.text = text;
		}
		
		@Override
		public void executeDo() {
			Location cl = cursorLocation;
			String curr = lines.isEmpty() ? "" : lines.remove(cl.getY());

			String pre = curr.substring(0, cl.getX());
			String post = curr.substring(cl.getX());
			
			if(!text.contains("\n")) {

				String newLine = "";
				newLine = pre + text + post;

				lines.add(cl.getY(), newLine);
				setCursorLocation(new Location(pre.length() + text.length(), cl.getY()));

			} else {
				String[] textArr = text.split("\n");
				int j = cl.getY();

				lines.add(j++, pre+textArr[0]);

				for(int i=1; i<textArr.length - 1; i++, j++) {
					lines.add(j, textArr[i]);
				}

				String last = textArr[textArr.length - 1];
				lines.add(j, last + post);
				setCursorLocation(new Location(last.length(), j));
			}
		}

		@Override
		public void executeUndo() {
			lines = listOld;
			cursorLocation = clOld;
		}
		
	}
}