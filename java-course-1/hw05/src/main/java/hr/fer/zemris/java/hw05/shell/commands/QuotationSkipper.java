package hr.fer.zemris.java.hw05.shell.commands;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.stream.Stream;

/**
 * Interface that parses text by skipping quotation marks
 * and by escaping some characters.
 * 
 * @author Ivan Rep
 */
public interface QuotationSkipper {

//	public static void main(String[] args) {
//		String s1 = "copy /home/john/info.txt /home/john/backupFolder";
//		String s2 = "copy 			\"C:/P\\\\rogram \\\"Files/Program1/info.txt\" \"C:/Program Files/Program1/info.txt\"";
//		String s3 = "ls .\\asdf.txt";
//		String s4 = "copy \"C:\\\\Documents and Settings\\\\Users\\\\javko\\\"\" ";
//		String s5 = "copy 			\"C:/P\\\\rogram \"Files/Program1/info.txt\" \"C:/Program Files/Program1/info.txt\"";
//		String s6 = "copy 			\"C:/P\\\\rogram \\\"Files/Program1/info.txt\"C:/ProgramFiles/Program1/info.txt";
//		String s7 = "copy 			\"C:/P\\\\rogram \\\"Files/Program1/info.txt\" C:/ProgramFiles/Program1/info.txt";
//		String s8 = " \"C:\\Users\\Ivan\\Desktop\\ gitlab(1).txt\"";
//		String s9 = "cat \"C:\\Users\\Ivan\\Desktop\\codeforces\\800\\team.cpp\" UTF-16";
//		
//		
//		System.out.println(Arrays.toString( skipQuotesAndSplit(s1) ));
//		System.out.println(Arrays.toString( skipQuotesAndSplit(s2) ));
//		System.out.println(Arrays.toString( skipQuotesAndSplit(s3) ));
//		System.out.println(Arrays.toString( skipQuotesAndSplit(s4) ));
//		System.out.println(Arrays.toString( skipQuotesAndSplit(s5) ));
//		System.out.println(Arrays.toString( skipQuotesAndSplit(s6) ));
//		System.out.println(Arrays.toString( skipQuotesAndSplit(s7) ));
//		System.out.println(Arrays.toString( skipQuotesAndSplit(s8) ));
//		System.out.println(Arrays.toString( skipQuotesAndSplit(s9) ));
//	}

	/**
	 * The method parses the given text.
	 * 
	 * @param text the text to be parsed
	 * @return an array of transformed strings from the given text
	 */
	static String[] skipQuotesAndSplit(String text) {

		int countQuotations = 0;
		text = text.trim();
		char[] c = text.toCharArray();
		ArrayList<String> strings = new ArrayList<>();

		int i, j;
		for(i=0, j=0; i<c.length; i++) {

			if(c[i]=='\"') countQuotations++;

			if(  c[i]==' ' || c[i]=='\t' ) {
				strings.add( text.substring(j, i++) );
				i = findNextNonBlank(i, c) - 1;
				j=i+1;
			}

			if(  c[i]=='\"' ) {
				int index = i+1;
				int[] arr = findEndOfQuotationAndEscapes(c, i);
				
				if(arr.length==0 ? true : c[ arr[arr.length-1] ]!='\"') return null;
				else countQuotations++;
				
				String add = "";

				if(arr.length!=0) 
					for(int k : arr) {
						add += text.substring(index, k);
						index=k+1;
					}
				if(index<c.length && c[index]!=' ') return null;
					
				strings.add(add);
				
				i = findNextNonBlank(arr[arr.length-1], c)+1;
				j=i;
			}


		}

		if(countQuotations%2!=0) return null;

		String end = text.substring(j, text.length());
		if( !end.isBlank() ) strings.add( end );

		String [] ret = new String[strings.size()];
		i = 0;
		for( String str : strings ) ret[i++] = str.trim();

		return ret;

	}

	/**
	 * Method finds the end of quotations marks and the indexes
	 * of the escape characters. Returns these values in the
	 * form of an array.
	 * 
	 * @param c an array of characters that is searched
	 * @param i the current index
	 * @return an array of indexes of escaped characters and the ending index
	 */
	public static int[] findEndOfQuotationAndEscapes(char[] c, int i) {

		ArrayList<Integer> esc = new ArrayList<>();

		for(int j=i+1; j<c.length; j++) 
			if(j+1<c.length && (c[j+1]=='\"' || c[j+1]=='\\') && c[j]=='\\') {
				esc.add(j);
				j+=1;
			} else if( c[j]=='\"' ) {
				esc.add(j);
				break;
			}


		int[] escArr = new int[esc.size()];
		for(int j=0; j<escArr.length; j++) 
			escArr[j]=esc.get(j);

		return escArr;
	}

	/**
	 * The method finds the index of the next non blank character
	 * in an array of characters starting from the current index.
	 * 
	 * @param i the current index
	 * @param c the array of characters
	 * @return the index of the next non blank character
	 */
	public static int findNextNonBlank(int i, char[] c) {
		while(  i<c.length  && (c[i]==' ' || c[i]=='\t') ) i++;
		return i;
	}

}
