package hr.fer.oprpp1.hw02.prob1;

/**
 * The class is a model of a lexical analyzer. 
 * 
 * @author Ivan Rep
 */
public class Lexer {

	private char[] data; 
	private Token token; 
	private int currentIndex; 
	private LexerState state;

	/**
	 * A constructor method that sets token to null, the state to LexerState.BASIC,
	 * the currentIndex to 0 and sets the value of the data array.
	 * 
	 * @param text the string that is converted to the data array
	 */
	public Lexer(String text) { 
		String str = stringWithoutBlankChars(text);
		data = text.equals("") ? text.toCharArray() : str.toCharArray();
		token = null;
		currentIndex = 0;
		state = LexerState.BASIC;
	}

	/**
	 * The method removes tabulators, carriage returns and new lines from the string. 
	 * It also replaces all multiple spaces with a single space.
	 * 
	 * @param text the whitespaces of this string are removed
	 * @return the string without multiple whitespaces
	 */
	private String stringWithoutBlankChars(String text) {
//		String s = text.replace("\r", " ").replace("\t", " ").replace("\n", " ").replaceAll(" +", " ").trim();
//		return s.equals(" ") ? "" : s;
		String s = text.replaceAll("\\s+", " ").trim();
		return s;
	}


	/**
	 * The method gets the next token from the input.
	 * 
	 * @return the next token
	 * @throws LexerException if there are no more tokens left
	 */
	public Token nextToken() {

		if( token!=null && token.getType().equals(TokenType.EOF) ) {
			throw new LexerException("There are no more tokens.");
		} else if( currentIndex >= data.length ) {
			token = new Token(TokenType.EOF, null);
		} else if( state.equals(LexerState.EXTENDED) ) {
			token = FindTokenInExtendedState(data);
		} else if( Character.isLetter( data[currentIndex] ) || data[currentIndex]=='\\'  ) {
			token = new Token( TokenType.WORD, getNextWord(data) );
		} else if( Character.isDigit( data[currentIndex] ) || data[currentIndex]=='\\' ) {
			token = new Token( TokenType.NUMBER, getNextNumber(data) );
		} else {
			token = new Token( TokenType.SYMBOL, getNextSymbol(data) );
		}

		return token; 
	}

	/**
	 * The method finds the next token in it's extended state.
	 * 
	 * @param data an array characters from the input 
	 * @return the next token in the extended state
	 */
	private Token FindTokenInExtendedState(char[] data) {
		if( data[currentIndex]==' ' )
			currentIndex++;
		Token t = null;
		switch (data[currentIndex]) {
		case '#' -> {
			t = new Token( TokenType.SYMBOL, '#' );
			currentIndex++;}
		default -> {
			int i, m;
			for(i=currentIndex, m=data.length; i<m; i++)
				if( data[i]=='#' || data[i]==' ' ) break;
			t = new Token( TokenType.WORD, new String(data).substring(currentIndex, i) );
			currentIndex = i;}
		}
		return t;

	}

	/**
	 * the method gets the next symbol from the input.
	 * 
	 * @param data an array of characters from the input
	 * @return an object of type Character
	 */
	private Character getNextSymbol(char[] data) {
		Character c = data[currentIndex++];
		if ( currentIndex < data.length-2 && data[currentIndex]==' ' )
			currentIndex++;
		return c;
	}

	/**
	 * the method gets the next number from the input.
	 * 
	 * @param data an array of characters from the input
	 * @return an object of type Long
	 */
	private Long getNextNumber( char[] data ) {
		int i, m;
		for(i=currentIndex, m=data.length; i<m; i++) 
			if( !Character.isDigit(data[i]) ) break;
		String s = new String(data).substring( currentIndex, i );
		Long l = null;
		try { l = Long.parseLong(s); } 
		catch(NumberFormatException exc) { throw new LexerException("The number "+s+" can't be parsed."); }
		currentIndex = ( i < data.length-2 && data[i]==' ' ) ? i+1 : i;
		return l;

	}

	/**
	 * the method gets the next word from the input.
	 * 
	 * @param data an array of characters from the input
	 * @return an object of type String
	 */
	private String getNextWord(char[] data ) {
		int i, size;
		StringBuilder sb = new StringBuilder();
		for( i=currentIndex, size=data.length; i<size; i++) {
			if( Character.isLetter(data[i]) );
			else if( data[i] == '\\' ) {
				if( ( i >= size-1 || Character.isLetter(data[i+1]) ) ) 
					throw new LexerException("There is a '\\\\' sign at the end of the input.");
				sb.append( new String(data).substring(currentIndex, i++) );
				currentIndex=i;
			}
			else break;
		}
		sb.append( new String(data).substring(currentIndex, i) );
		currentIndex = ( i < data.length-2 && data[i]==' ' ) ? i+1 : i;
		return sb.toString();
	}

	/**
	 * The method returns the current token from the Lexer object.
	 * 
	 * @return the current token
	 */
	public Token getToken() {
		return this.token;
	}

	/**
	 * The method sets the state of the LexerState variable.
	 * 
	 * @param state the value that the LexerState variable is set to.
	 * @throws NullPointerException if the state is <code>null</code>
	 */
	public void setState(LexerState state) {
		if(state==null) throw new NullPointerException("The state cannot be null.");
		this.state = state;
	}

}