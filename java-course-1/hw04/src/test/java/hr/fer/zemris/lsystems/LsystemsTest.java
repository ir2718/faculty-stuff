package hr.fer.zemris.lsystems;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

import hr.fer.oprpp1.custom.collections.Dictionary;

public class LsystemsTest {
	
	@Test
	public void TestGenerateZero() {
		Dictionary<Character, String> productions = CreateAndfillProductions();
		assertEquals( "F", generate( 0, "F", productions) );
	}

	
	@Test
	public void TestGenerateOne() {
		Dictionary<Character, String> productions = CreateAndfillProductions();
		assertEquals( "F+F--F+F", generate( 1, "F", productions) );
	}
	
	
	@Test
	public void TestGenerateTwo() {
		Dictionary<Character, String> productions = CreateAndfillProductions();
		assertEquals( "F+F--F+F+F+F--F+F--F+F--F+F+F+F--F+F", generate( 2, "F", productions) );
	}
	
	@Test
	public void testinjos() {
		
				
		Dictionary<Character, String> productions = new Dictionary<>();
		productions.put('F', "FF+[+F-F-F]-[-F+F+F]");
		
		assertEquals( "GFF+[+F-F-F]-[-F+F+F]FF+[+F-F-F]-[-F+F+F]+[+FF+[+F-F-F]-[-F+F+F]-FF+[+F-F-F]-[-F+F+F]-FF+[+F-F-F]-[-F+F+F]]-[-FF+[+F-F-F]-[-F+F+F]+FF+[+F-F-F]-[-F+F+F]+FF+[+F-F-F]-[-F+F+F]]"
				, generate( 2, "GF", productions) );
		
	}
	

	private Dictionary<Character,String> CreateAndfillProductions() {
		Dictionary<Character, String> productions = new Dictionary<>();
		productions.put('F', "F+F--F+F");
		return productions;
	}	


	public String generate(int arg0, String axiom, Dictionary<Character, String> productions) {
		if(arg0==0) return axiom;

		char[] helper = axiom.toCharArray();
		StringBuilder sb = new StringBuilder();

		int j, k;
		for( int i=0; i < arg0; i++ ) {

			j=k=0;
			for( int m=helper.length; j<m ; j++) 
				if( productions.get( helper[j] ) != null ) {
					sb.append( new String(helper).substring(k, j) )
					.append( productions.get(helper[j]) );
					k=j+1;
				}
			sb.append(new String(helper).substring(k, helper.length) );

			helper = sb.toString().toCharArray();
			if(i!=arg0-1) sb = new StringBuilder();
		}

		return sb.toString();

	}

}
