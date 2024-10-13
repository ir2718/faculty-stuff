package hr.fer.oprpp1.hw04.db;

import java.util.ArrayList;
import java.util.List;

/**
 * The class represents a parser used in parsing a database query.
 * 
 * @author Ivan Rep
 */
public class QueryParser {

	private QueryLexer lexer;
	private ArrayList<QueryLexerElement> arrElems;

	/**
	 * The constructor method for the parser.
	 * Takes in the body of the query and 
	 * parses all the elements.
	 * 
	 * @param query the body of the query
	 */
	public QueryParser(String query) {
		
		lexer = new QueryLexer(query.trim());
		arrElems = new ArrayList<>();

		QueryLexerElement el = null;
		do {
			el = lexer.nextElem();
			arrElems.add(el);
		}while( !el.getType().equals(QueryLexerConstant.EOF) );
	}

	/**
	 * Checks if the query is in the format <code>jmbag = "xxxxxxxxxx"<\code>.
	 * 
	 * @return true if it is, otherwise false
	 */
	public boolean isDirectQuery() {

		return arrElems.size()==4 
				&& arrElems.get(0).getType().equals(QueryLexerConstant.JMBAG)
				&& arrElems.get(1).getValue().equals("=")
				&& arrElems.get(2).getType().equals(QueryLexerConstant.TEXT)
				&& arrElems.get(3).getType().equals(QueryLexerConstant.EOF);
	}

	/**
	 * Returns the jmbag contained in the query.
	 * 
	 * @return the string representation of the jmbag parameter in the query
	 * @throws IllegalStateException if the query is not a direct query
	 */
	public String getQueriedJMBAG() {
		if( !this.isDirectQuery() ) throw new IllegalStateException("The query must be direct in order to use this method.");
		return arrElems.get(2).getValue();
	}

	/**
	 * Returns a list of conditions contained in the query.
	 * 
	 * @return a list of conditions contained in the query
	 */
	public List<ConditionalExpression> getQuery() {
		int i = 0;
		List<ConditionalExpression> listOfExpression = new ArrayList<>();
		while( !arrElems.get(i).getType().equals(QueryLexerConstant.EOF ) ) {

			if( arrElems.get(i).getType().equals(QueryLexerConstant.LOGICAL) )i++;

			if( !IsGoodConditionalExpression(i, arrElems) ) 
				throw new IllegalArgumentException("The query isn't formed correctly.");
			
			IFieldValueGetter field = null;
			switch( arrElems.get(i).getValue() ) {
			case 	"jmbag" 	-> field = FieldValueGetters.JMBAG;
			case 	"firstName"	-> field = FieldValueGetters.FIRST_NAME;
			case 	"lastName" 	-> field = FieldValueGetters.LAST_NAME;
			}

			IComparisonOperator comp = null;
			switch( arrElems.get(i+1).getValue() ) {
			case "<" 	-> comp = ComparisonOperators.LESS;
			case "<=" 	-> comp = ComparisonOperators.LESS_OR_EQUALS;
			case "=" 	-> comp = ComparisonOperators.EQUALS;
			case ">=" 	-> comp = ComparisonOperators.GREATER_OR_EQUALS;
			case ">" 	-> comp = ComparisonOperators.GREATER;
			case "!=" 	-> comp = ComparisonOperators.NOT_EQUALS;
			case "LIKE" -> comp = ComparisonOperators.LIKE;
			}

			String literal = arrElems.get(i+2).getValue();
			if( arrElems.get(i).getType().equals(QueryLexerConstant.JMBAG) && ( literal.length()!=10 || !ContainsOnlyDigits(literal) ) )
				throw new IllegalStateException("The jmbag value must be 10 characters long and all characters must be digits.");
			
			ConditionalExpression cond = new ConditionalExpression(field, literal, comp);
			listOfExpression.add(cond);
			i+=3;

		}

		return listOfExpression;
	}


	/**
	 * The method checks whether the given string contains 
	 * only digits. 
	 * 
	 * @param literal the given string
	 * @return true if there are only digits in the string, otherwise false
	 */
	private boolean ContainsOnlyDigits(String literal) {
		char[] stringArr = literal.toCharArray();
		
		for( char c : stringArr )
			if( !Character.isDigit(c) )
				return false;
		
		return true;
		
	}
	
	/**
	 * Returns true if the expression contains 
	 * an identificator, a relational operator
	 * and text.
	 * 
	 * @param i the index of the identificator in arrElems list
	 * @param arrElems the list with the elements 
	 * @return true if the expression consists of an identificator,
	 * a relational operator and text, otherwise false
	 */
	private static boolean IsGoodConditionalExpression(int i, ArrayList<QueryLexerElement> arrElems) {
		return ( arrElems.get(i).getType().equals(QueryLexerConstant.JMBAG) 
				|| arrElems.get(i).getType().equals(QueryLexerConstant.IDN) ) 
		&& arrElems.get(i+1).getType().equals(QueryLexerConstant.RELATIONAL)
		&& arrElems.get(i+2).getType().equals(QueryLexerConstant.TEXT);
	}

}
