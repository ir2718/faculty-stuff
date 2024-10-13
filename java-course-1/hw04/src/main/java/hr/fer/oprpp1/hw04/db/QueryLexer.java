package hr.fer.oprpp1.hw04.db;

/**
 * The class represents a lexer for the database query.
 * 
 * @author Ivan Rep
 */
public class QueryLexer {

	private char[] query;
	private QueryLexerElement current;
	private int currentIndex;

	/**
	 * The constructor method that takes in the query body.
	 * 
	 * @param query the given query's body
	 */
	public QueryLexer(String query) {
		this.query = query.trim().toCharArray();
		currentIndex = 0;
		current = null;
	}

	/**
	 * The method returns the next element in the 
	 * process of lexing the given input.
	 * 
	 * @return the next element in the process of lexing
	 */
	public QueryLexerElement nextElem() {

		findNextNonBlank();

		if ( currentIndex>=query.length )
			current = new QueryLexerElement( QueryLexerConstant.EOF,  null );
		else if( IsJmbag() ) 
			current = new QueryLexerElement( QueryLexerConstant.JMBAG, getJmbag() );
		else if ( IsIdn() ) 
			current = new QueryLexerElement( QueryLexerConstant.IDN, getIdn() );
		else if ( IsRelational() ) 
			current = new QueryLexerElement( QueryLexerConstant.RELATIONAL, getRelational() );
		else if ( IsLogical() ) 
			current = new QueryLexerElement( QueryLexerConstant.LOGICAL, getLogical() );
		else if ( IsText() )
			current = new QueryLexerElement( QueryLexerConstant.TEXT, getText() );

		return current;

	}

	/**
	 * Returns the "jmbag" constant and updates the current index
	 * in the process of lexing the next element.
	 * 
	 * @return the constant "jmbag"
	 */
	private String getJmbag() {
		currentIndex+=5;
		return "jmbag";
	}

	/**
	 * Returns the part of the query in quotation marks
	 * and updates the current index.
	 * Used in getting the next element while lexing.
	 * 
	 * @return the text in the quotation marks
	 */
	private String getText() {
		int i=currentIndex;
		if(query[i]=='\"')i++;
		
		for(int m=query.length; i<m; i++) 
			if(query[i]=='\"')  break;

		int copyOfCurrentIndex = currentIndex;
		currentIndex = i+1;
		return new String(query).substring(copyOfCurrentIndex+1, i);
	}

	/**
	 * Checks if a certain part of the query is text
	 * contained in the quotation marks.
	 * 
	 * @return true if it is in quotation marks, otherwise false
	 * @throws IllegalArgumentException if the quotation marks aren't closed
	 */
	private boolean IsText() {
		int i=currentIndex;

		if(query[i]=='\"')i++;
		else throw new IllegalArgumentException("A word is incorrectly spelled or the text isn't surrounded by quotation marks.");

		for(int m=query.length; i<m; i++) 
			if(query[i]=='\"')  break;

		if(i==query.length) throw new IllegalArgumentException("The quotation marks aren't closed.");
		return true;
	}

	/**
	 * Returns the string constant "AND" and updates the current index.
	 * 
	 * @return the string constant "AND"
	 */
	private String getLogical() {
		currentIndex+=3;
		return "AND";
	}

	/**
	 * Returns the next relational operator and updates the current index.
	 * 
	 * @return the string that represents the current operator
	 */
	private String getRelational() {
		int i=currentIndex;

		if ( i+1<query.length && query[i]=='<' && query[i+1]=='=' ) {
			currentIndex+=2;
			return "<=";
		}  else if ( i+1<query.length && query[i]=='>' && query[i+1]=='=' ) {
			currentIndex+=2;
			return ">=";
		} else if ( query[i]=='=' ) {
			currentIndex++;
			return "=";
		} else if ( query[i]=='>' ) {
			currentIndex++;
			return ">";
		} else if ( query[i]=='<' ) {
			currentIndex++;
			return "<";
		} else if ( i+1<query.length && query[i]=='!' && query[i+1]=='=' ) {
			currentIndex+=2;
			return "!=";
		} else if ( i+3<query.length && new String(query).substring(i, i+4).toUpperCase().equals("LIKE") ) {
			currentIndex+=4;
			return "LIKE";
		} 

		currentIndex++;
		return "<";
	}

	/**
	 * Returns the constant "lastName" or "firstName"
	 * and updates the current index.
	 * 
	 * @return the constant "lastName" or "firstName"
	 */
	private String getIdn() {
		int copyOfCurrentIndex = currentIndex;
		if ( currentIndex+7<query.length && query[currentIndex]=='l' ) {
			currentIndex+=8;
			return new String(query).substring(copyOfCurrentIndex, currentIndex);
		}
		currentIndex+=9;
		return new String(query).substring(copyOfCurrentIndex, currentIndex);
	}

	/**
	 * Checks if the next element is a logical operator "and".
	 * It's case insensitive.
	 * 
	 * @return true if it is "and", false otherwise
	 */
	private boolean IsLogical() {
		int i=currentIndex;
		if( i+3<query.length && new String(query).substring(i, i+3).toUpperCase().equals("AND") ) return true;
		return false;
	}

	/**
	 * Checks if the next element is a relational operator.
	 * 
	 * @return true if it is a relational operator, otherwise false
	 */
	private boolean IsRelational() {
		int i=currentIndex;

		if ( i+1<query.length && query[i]=='<' && query[i+1]=='=' ) 
			return true;
		else if ( i+1<query.length && query[i]=='>' && query[i+1]=='=' ) 
			return true;
		else if ( i+1<query.length &&  query[i]=='!' && query[i+1]=='=' ) 
			return true;
		else if ( i+3<query.length && new String(query).substring(i, i+4).toUpperCase().equals("LIKE") ) 
			return true;
		else if ( query[i]=='<' || query[i]=='>' || query[i]=='=' )
			return true;

		return false;
	}

	/**
	 * Checks whether the next element is "lastName" or "firstName".
	 * 
	 * @return true if is, otherwise false
	 */
	private boolean IsIdn() {
		int i=currentIndex; 
		if( i+7<query.length && query[i]=='l' )
			return new String(query).substring(i, i+8).equals("lastName");
		else if( i+8<query.length && query[i]=='f' )
			return new String(query).substring(i, i+9).equals("firstName");
		return false;
	}

	/**
	 * Checks whether the next element is "jmbag".
	 * 
	 * @return true if is, otherwise false
	 */
	private boolean IsJmbag() {
		if ( currentIndex+5>=query.length || query[currentIndex]!='j' ) return false; 
		return new String(query).substring(currentIndex, currentIndex+5).equals("jmbag");
	}

	/**
	 * The method skips all whitespaces and updates the current index
	 * to the first non blank.
	 */
	private void findNextNonBlank() {
		int i = currentIndex;
		
		for(int m=query.length; i<m; i++) 
			if( query[i]!=' ' &&  query[i]!='\t' ) 
				break;
		
		currentIndex = i;
	}

}
