package hr.fer.oprpp1.hw04.db;

import java.util.List;

/**
 * The class represents a filter for the database query.
 * 
 * @author Ivan Rep
 */
public class QueryFilter implements IFilter{
	
	private List<ConditionalExpression> list;
	
	/**
	 * The constructor for the query filter.
	 * Takes in the list of conditional expressions 
	 * contained in the query.
	 * 
	 * @param list the list of conditional expressions fromt the query
	 */
	public QueryFilter(List<ConditionalExpression> list) {
		this.list = list;
	}
	
	/**
	 * The method checks if one of the given record's attributes
	 * and string literal satisfy the comparison operator.
	 * 
	 * @param record the given student record
	 * @return true if satisfied, otherwise false
	 */
	@Override
	public boolean accepts(StudentRecord record) {
		for( ConditionalExpression exc : list ) 
			if( !exc.getComparisonOperator()
					.satisfied( exc.getFieldGetter().get(record), exc.getStringLiteral() ) )
				return false;
		
		return true;
	}

}
