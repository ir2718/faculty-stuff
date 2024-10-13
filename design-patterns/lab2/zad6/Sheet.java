package zad6;

import java.util.LinkedList;
import java.util.List;

import static java.util.Arrays.stream;

public class Sheet {

	private Cell[][] arr2d;

	public Sheet(int row, int col) {
		this.arr2d = new Cell[row][col];

		for(int i=0; i<row; i++) 
			for(int j=0; j<col; j++) 
				arr2d[i][j] = new Cell(this);
	}

	public Cell cell(String ref) {
		int row = (int) (ref.charAt(0)) - 'A'; // ascii 65 = A
		int col = (int) (ref.charAt(1)) - '1'; // ascii 49 = 1
		return this.arr2d[row][col];
	}

	public void set(String ref, String content) {
		Cell cell = cell(ref);
		List<Cell> refs; 

		this.checkForCycle(cell, content);

		refs = getRefs(cell);
		for(Cell c : refs)
			c.removeListener(cell);

		cell.setExp(content);
		cell.setValue(evaluate(cell));

		refs = getRefs(cell);
		for(Cell c : refs) 
			c.addListener(cell);

		cell.setCoordinate(ref);
		cell.notifyListeners();
	}

	private void checkForCycle(Cell c, String content) {

		List<Cell> open = new LinkedList<>();

		if (content.contains("+")) {
			String[] coords = content.split("\\+");
			for(String s : coords) 
				open.add(cell(s));
		} else if (Character.isAlphabetic(content.charAt(0))) {
			open.add(cell(content));
		} else {
			return;
		}

		open.add(c);
		int count = 0;

		while(!open.isEmpty()) {
			Cell cell = open.remove(0);

			if(cell.equals(c) && count==0)
				count++;
			else if(cell.equals(c) && count==1)
				throw new RuntimeException("Circular reference found.\n");

			List<Cell> expand = getRefs(cell);
			if(!expand.isEmpty()) open.addAll(0, expand);
		}
	}

	public List<Cell> getRefs(Cell cell) {
		String exps = cell.getExp();
		List<Cell> cells = new LinkedList<>(); 

		if(!Character.isAlphabetic(exps.charAt(0))) 
			return cells;

		if(!exps.contains("+")) {
			List<Cell> retVal = new LinkedList<>();
			retVal.add(cell(exps));
			return retVal;
		}

		String[] refs = exps.split("\\+");
		stream(refs).forEach(exp -> cells.add(cell(exp)));

		return cells;
	}

	public int evaluate(Cell cell) {
		int retVal = 0;

		char[] cArr = cell.getExp().toCharArray();

		if(Character.isAlphabetic(cArr[0])) 
			retVal = getRefs(cell).stream().mapToInt(c -> c.getValue()).sum();
		else
			retVal = Integer.parseInt(cell.getExp());

		return retVal;
	}

	@Override
	public String toString() {

		StringBuilder sb = new StringBuilder();

		for(int i=0; i<arr2d.length; i++) {
			for(int j=0; j<arr2d.length; j++)	
				sb.append(arr2d[i][j]).append(" ");
			sb.append("\n");
		}

		return sb.toString();
	}

}
