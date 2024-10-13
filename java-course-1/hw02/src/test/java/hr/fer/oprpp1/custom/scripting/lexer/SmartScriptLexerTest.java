package hr.fer.oprpp1.custom.scripting.lexer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

import hr.fer.oprpp1.custom.scripting.elems.Element;
import hr.fer.oprpp1.custom.scripting.elems.ElementString;
import hr.fer.oprpp1.hw02.prob1.Lexer;

public class SmartScriptLexerTest {

	@Test
	public void testNotNull() {
		SmartScriptLexer lexer = new SmartScriptLexer("");
		
		assertNotNull(lexer.nextToken(), "Token was expected but null was returned.");
	}

	@Test
	public void testNullInput() {
		assertThrows(NullPointerException.class, () -> new SmartScriptLexer(null));
	}

	@Test
	public void testSimpleInputTag() {
	
		String input = "This is sample text. {$= abcdefgh abc $}";
		SmartScriptLexer l = new SmartScriptLexer(input);
		
		assertEquals( "This is sample text. ", l.nextToken().asText() );
		l.nextToken();
		assertEquals( "abcdefgh", l.nextToken().asText() );
		assertEquals( "abc", l.nextToken().asText() );
		
	}
	
	@Test
	public void testComplicatedInputTag() {
	
		String input = "This is sample text. {$      =           xyzxy33z 	 \n $}";
		SmartScriptLexer l = new SmartScriptLexer(input);
		
		String s1 = l.nextToken().asText();
		assertEquals( "This is sample text. ", s1, "The expected output was This is sample text, "
				+ "but it was "+s1 );
		l.nextToken();
		String s2 = l.nextToken().asText();
		assertEquals( "xyzxy33z", s2, "The expected output was xyzxy33z, "
				+ "but it was "+s2 );
		
	}
	
	
	@Test
	public void testMoreComplicatedInputTagThrowsException() {
	
		
		String input="This is sample txt. "
				+ "{$ =  asdf asdf      asdf            asdf           }";
		
		SmartScriptLexer l = new SmartScriptLexer(input);
		
		assertEquals( "This is sample txt. ", l.nextToken().asText() );
		assertThrows( SmartScriptLexerException.class, ()->l.nextToken(), "The test was expecting a SmartScriptLexerException"
				+ " but there that exception wasn't thrown." ) ;
		
	}
	
	@Test
	public void testPartOfTheInputFromTheHomeworkPDF() {
	
		
		String input = "This is sample text.\r\n"
				+ "{$ FOR i 1 10 1 $}\r\n"
				+ " This is {$= i $}-th time this message is generated.\r\n"
				+ "{$END$}\r\n";
		
		SmartScriptLexer l = new SmartScriptLexer(input);
		
		assertEquals( "This is sample text.\r\n", l.nextToken().asText() );
		
		assertEquals( "FOR", l.nextToken().asText() );
		assertEquals( "i", l.nextToken().asText() );
		assertEquals( "1", l.nextToken().asText() );
		assertEquals( "10", l.nextToken().asText() );
		assertEquals( "1", l.nextToken().asText() );
		l.nextToken();
		
		assertEquals( "This is", l.nextToken().asText().trim() );
		
		assertEquals( "ECHO", l.nextToken().asText() );
		assertEquals( "i", l.nextToken().asText() );
		l.nextToken();
		assertEquals( "-th time this message is generated.", l.nextToken().asText().trim());
		
		assertEquals( "END", l.nextToken().asText() );
		
		
	}
	
	
	
	@Test
	public void testTwoForLoops() {
	
		
		String input =
				  "{$ FOR i 1 10 1 $}" 
				+ " This is {$= i $}-th time this message is generated.\n"
				+ "{$END$}"
				+ "{$     FOR 2      34234 234234 23423.4 $}" 
				+ " This is {$= i $}-th time this message is generated.\r\n"
				+ "{$END$}";
		
		SmartScriptLexer l = new SmartScriptLexer(input);
		
		assertEquals( "FOR", l.nextToken().asText() );
		assertEquals( "i", l.nextToken().asText() );
		assertEquals( "1", l.nextToken().asText() );
		assertEquals( "10", l.nextToken().asText() );
		assertEquals( "1", l.nextToken().asText() );
		l.nextToken();
		
		assertEquals( "This is", l.nextToken().asText().trim() );
		
		assertEquals( "ECHO", l.nextToken().asText() );
		assertEquals( "i", l.nextToken().asText() );
		l.nextToken();
		assertEquals( "-th time this message is generated.", l.nextToken().asText().trim());
		
		assertEquals( "END", l.nextToken().asText() );
		l.nextToken();
		assertEquals( "FOR", l.nextToken().asText() );
		assertEquals( "2", l.nextToken().asText() );
		assertEquals( "34234", l.nextToken().asText() );
		assertEquals( "234234", l.nextToken().asText() );
		assertEquals( "23423.4", l.nextToken().asText() );
		l.nextToken();
		
		assertEquals( "This is", l.nextToken().asText().trim() );
		
		assertEquals( "ECHO", l.nextToken().asText() );
		assertEquals( "i", l.nextToken().asText() );
		l.nextToken();
		assertEquals( "-th time this message is generated.", l.nextToken().asText().trim());
		l.nextToken();
		
		
	}

	
	@Test
	public void testThrowsBecauseOfEscapeCharacters() {
	
		
		String input = "This is 1st tim\\e this message is generated.\n";
		
		SmartScriptLexer l = new SmartScriptLexer(input);
		
		assertThrows(SmartScriptLexerException.class, () -> l.nextToken(), "It was"
				+ " expected that a SmartScriptLexerException would be thrown, but it "
				+ "wasn't" );
		
	}
	
	@Test
	public void testNoSpacesInFor() {
	
		
		String input = "{$ FOR i-1.35bbb\"1 asdf 2\" $}";
		
		SmartScriptLexer l = new SmartScriptLexer(input);
		
		assertEquals( "FOR", l.nextToken().asText() );
		assertEquals( "i", l.nextToken().asText() );
		assertEquals( "-1.35", l.nextToken().asText() );
		assertEquals( "bbb", l.nextToken().asText() );
		assertEquals( "1 asdf 2" , l.nextToken().asText() );
		
	}
	
	
	

	
}
