package hr.fer.oprpp1.custom.scripting.lexer;

import hr.fer.oprpp1.custom.scripting.elems.*;

/**
 * The class represents a lexer for the language.
 * 
 * @author Ivan Rep
 */
public class SmartScriptLexer {

	private char[] documentBody;
	private Element elem;
	private int currentIndex;
	private SmartScriptLexerState state;

	/**
	 * The getter method for the current state of the lexer.
	 * 
	 * @return a SmartScriptLexerState enum: TAGSTATE or TEXTSTATE
	 */
	public SmartScriptLexerState getState() {
		return this.state;
	}

	/**
	 * The getter method for the index of the first character that hasn't been lexically analysed yet.
	 * 
	 * @return the index of the first character that hasn't been analysed
	 */
	public int getCurrentIndex() {
		return this.currentIndex;
	}

	/**
	 * The getter method for the length of the given document.
	 * 
	 * @return the length of the given document.
	 */
	public int getDocumentLength() {
		return documentBody==null ? 0 : this.documentBody.length;
	}

	/**
	 * The constructor method of the SmartScriptLexer class.
	 * 
	 * @param documentBody the string to be lexically analysed
	 */
	public SmartScriptLexer(String documentBody) {
		this.documentBody = documentBody.toCharArray(); // stringWithoutBlankChars(documentBody).toCharArray();
		elem = null;
		currentIndex = 0;
		state = SmartScriptLexerState.TEXTSTATE;
	}

	/**
	 * The method finds the next next token to be parsed in the given input.
	 * 
	 * @return the next element that is found in the given input
	 * @throws SmartScriptLexerException if any tag is written incorrectly
	 */
	public Element nextToken() {
		
		currentIndex = IndexOfNextNonBlank(currentIndex);

		if ( currentIndex+1 < documentBody.length && state.equals( SmartScriptLexerState.TAGSTATE ) && 
				documentBody[currentIndex]=='$' && documentBody[currentIndex+1]=='}' ) {
			state=SmartScriptLexerState.TEXTSTATE;
			currentIndex += 2;
			elem = new ElementString("$}");
			
		} else if ( currentIndex>=documentBody.length ) {
			elem = new ElementString("");
			
		} else if ( currentIndex+1 < documentBody.length && state.equals( SmartScriptLexerState.TEXTSTATE ) &&
				documentBody[currentIndex]=='{' && documentBody[currentIndex+1]=='$') { 
			if( !isTagClosedOrNested() ) throw new SmartScriptLexerException("The tag isn't written correctly.");
			elem = new ElementString( findTagType() );
			state = SmartScriptLexerState.TAGSTATE;
			currentIndex = IndexOfNextNonBlank(currentIndex);
			
		} else if( state.equals( SmartScriptLexerState.TEXTSTATE ) ) {
			elem = new ElementString( findText() );
			currentIndex = IndexOfNextNonBlank(currentIndex);
			
		} else if( state.equals( SmartScriptLexerState.TAGSTATE ) ) {
			elem = findNextElement();
			currentIndex = IndexOfNextNonBlank(currentIndex);
		}


		return elem;
	}

	/**
	 * The method is used for finding the next element in the current tag.
	 * 
	 * @return the next element in the current tag
	 * @throws SmartScriptParseLexerException if a number is parsed and it has more than 2 dots
	 */
	private Element findNextElement() {
		int i;
		int m = documentBody.length;
		int copyOfCurrentIndex=currentIndex;
		Element retElem = null;

		if( Character.isLetter( documentBody[currentIndex] ) ) {
			for(i=currentIndex+1; i<m; i++)
				if( !Character.isLetter(documentBody[i]) && documentBody[i]!='_' && !Character.isDigit(documentBody[i]) )
					break;
			currentIndex=i;
			retElem = new ElementVariable( new String(documentBody).substring(copyOfCurrentIndex, currentIndex) );
			
		} else if ( currentIndex+1<m && documentBody[currentIndex]=='@' && Character.isLetter( documentBody[currentIndex+1] ) ) {
			for(i=currentIndex+2; i<m; i++)
				if( !Character.isLetter(documentBody[i]) && documentBody[i]!='_' && !Character.isDigit(documentBody[i]) )
					break;
			currentIndex=i;
			retElem =  new ElementFunction( new String(documentBody).substring(copyOfCurrentIndex, currentIndex) );
		
		} else if ( documentBody[currentIndex]=='+' || documentBody[currentIndex]=='/' || 
				documentBody[currentIndex]=='*' || documentBody[currentIndex]=='^' ) {
			currentIndex++;
			retElem =  new ElementOperator( new String(documentBody).substring(copyOfCurrentIndex, currentIndex) );
		
		} else if ( documentBody[currentIndex]=='-' && documentBody[currentIndex+1]==' ' ) {
			currentIndex++;
			retElem =  new ElementOperator( new String(documentBody).substring(copyOfCurrentIndex, currentIndex) );
		
		} else if(  (currentIndex+1<m && documentBody[currentIndex]=='-' && Character.isDigit( documentBody[currentIndex+1] ))
				|| Character.isDigit( documentBody[currentIndex] )) {
			int countDot=0;
			for(i=currentIndex+1; i<m; i++)
				if( Character.isDigit( documentBody[i] ) );
				else if( documentBody[i]=='.' ) countDot++;
				else break;

			currentIndex = i;

			if( countDot==0 )
				retElem =  new ElementConstantInteger( Integer.parseInt( new String(documentBody).substring( copyOfCurrentIndex, i ) ) );
			else if( countDot==1 )
				retElem =  new ElementConstantDouble( Double.parseDouble( new String(documentBody).substring( copyOfCurrentIndex, i ) ) );
			else throw new SmartScriptLexerException("A number can't be parsed because it has too many dots.");
		
		} else if ( documentBody[currentIndex]=='\"' ) {
			for( i = currentIndex+1; i+1 < m && documentBody[i]!='$' && documentBody[i+1]!='}'; i++ )
				if( documentBody[i]=='\\' && ( documentBody[i+1]=='\"' || documentBody[i+1]=='\\' ) ) i++;
				else if( documentBody[i]=='\"' )break;
			currentIndex = i+1;
			retElem =  new ElementString( new String(documentBody).substring(copyOfCurrentIndex+1, i) );
		}
		
		return retElem;

	}

	
	/**
	 * The method finds the type of the next tag from the given input. The possible outcomes
	 * are "FOR", "END" or "ECHO".
	 * 
	 * @return the string that corresponds to the tag
	 * @throws SmartScriptLexerExeption if the tag type doesn't match one of the given three
	 */
	private String findTagType() {
		int i = new String(documentBody).substring(currentIndex).indexOf('$')+1+currentIndex, m = documentBody.length;
		for( ; i < m ; i++)
			if( documentBody[i]!=' ' ) break;

		if ( i+3<m && Character.toUpperCase(documentBody[i])=='F' &&  Character.toUpperCase(documentBody[i+1])=='O' 
				&&  Character.toUpperCase(documentBody[i+2])=='R' && documentBody[i+3]==' ' ) {
			currentIndex = IndexOfNextNonBlank(i+3);
			return "FOR";
			
		} else if ( i+2<m &&  Character.toUpperCase(documentBody[i])=='E' && Character.toUpperCase(documentBody[i+1])=='N' 
				&&  Character.toUpperCase(documentBody[i+2])=='D' ) {
			currentIndex= IndexOfNextNonBlank(i+3);
			return "END";
			
		} else if ( documentBody[i]=='=' ) {
			currentIndex = IndexOfNextNonBlank(i+1);
			return "ECHO";
		}

		throw new SmartScriptLexerException("The first word in the tag doesn't match END, FOR or the equality sign.");
	}
	
	
	/**
	 * The method finds the first non blank char in the documentBody array starting from the given index.
	 * 
	 * @param i the starting index
	 * @return the index of the first non blank char
	 */
	private int IndexOfNextNonBlank(int i) {
		for(int m=documentBody.length; i<m; i++ )
			if(documentBody[i]!=' ') 
				break;
		return i;
	}
 

	/**
	 * The method checks if the tags are nested, if they're properly ended or if there are an uneven amount of quotation marks 
	 * that aren't escaped.
	 * 
	 * @return true if all conditions are fulfilled, otherwise false
	 * @throws SmartScriptParseLexerException if the tags are nested, if they're properly ended or if there are an uneven amount of quotation marks
	 */
	private boolean isTagClosedOrNested() {
		int i, m;
		int countQuotation=0;
		if( documentBody!=null && ( currentIndex+1<documentBody.length && ( documentBody[currentIndex]!='{' || documentBody[currentIndex+1]!='$' ) ) ) 
			return false;
		
		for( i = currentIndex+2, m = documentBody.length ; i<m; i++ ) {
			if( i+1<m && documentBody[i]=='$' && documentBody[i+1]=='}' ) break;
			else if (documentBody[i]=='\"') countQuotation++;
			else if ( i+1<m && documentBody[i]=='\\' && documentBody[i+1]=='\"') i++;
			else if( documentBody[i]=='{' && documentBody[i+1]=='$' ) throw new SmartScriptLexerException("Tags can't be nested");
		}
		
		if(countQuotation%2!=0) throw new SmartScriptLexerException("One of the quotation marks isn't closed.");
		if(i==m) throw new SmartScriptLexerException("A tag isn't closed");
		
		return documentBody[i]=='$' && documentBody[i+1]=='}' ? true : false;
	}


	/**
	 * The method finds the string required for the ElementString object from the given input.
	 * It begins the search from the current index and skips over "\\" and "\{".
	 * 
	 * @return a string of text that defines the next element
	 * @throws SmartScriptLexerException if the escaped characters aren't "\\" or "\{"
	 */
	private String findText() {
		int i, m;
		int copyOfCurrentIndex;
		
		for(i=currentIndex, m=documentBody.length; i<m; i++ ) {
			if ( i+1<m && documentBody[i]=='{' && documentBody[i+1]=='$' ) {
				break;
				
			} else if ( i+1<m && documentBody[i]=='\\' ) {
				if ( documentBody[i+1]=='\\' || documentBody[i+1]=='{' ) 
					i++;
				else 
					throw new SmartScriptLexerException("The only allowed escape characters are \\\\ and \\{");
				
			} else if ( i+1==m && documentBody[i]=='\\' )
				throw new SmartScriptLexerException("The only allowed escape characters are \\\\ and \\{");
		}
		
		copyOfCurrentIndex = currentIndex;
		currentIndex=i;
		return new String(documentBody).substring( copyOfCurrentIndex, i );
	}

}