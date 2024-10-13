package hr.fer.oprpp1.custom.scripting.parser;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;


public class SmartScriptParserTest {

	@Test
	public void testForShouldThrowNumberInsteadOfVariableName() {
		
		String s = "{$ FOR 3 1 10 1 $}";
		
		assertThrows( SmartScriptParserException.class, () -> new SmartScriptParser(s), "The test should throw a SmartScriptParserException, but it doesn't" );  
	}

	
	
	@Test
	public void testForShouldThrowSymbolInsteadOfVariableName() {
		
		String s = "{$ FOR * \"1\" -10 \"1\" $}";
		
		assertThrows( SmartScriptParserException.class, () -> new SmartScriptParser(s), "The test should throw a SmartScriptParserException, but it doesn't" );  
	}

	
	@Test
	public void testForShouldThrowFunctionInsteadOfNumberOrVariableName() {
		
		String s = "{$ FOR year @sin 10 $}";
		
		assertThrows( SmartScriptParserException.class, () -> new SmartScriptParser(s), "The test should throw a SmartScriptParserException, but it doesn't" );  
	}
	
	
	@Test
	public void testForShouldThrowTooManyArguments() {
		
		String s = "{$ FOR year 1 10 \"1\" \"10\" $}";
		
		assertThrows( SmartScriptParserException.class, () -> new SmartScriptParser(s), "The test should throw a SmartScriptParserException, but it doesn't" );  
	}
	
	

	@Test
	public void testForShouldThrowTooLittleArguments() {
		
		String s = "{$ FOR year $}";
		
		assertThrows( SmartScriptParserException.class, () -> new SmartScriptParser(s), "The test should throw a SmartScriptParserException, but it doesn't" );  
	}
	
	
	@Test
	public void testForShouldThrowTooManyArgumentsAgain() {
		
		String s = "{$ FOR year 1 10 1 3 $} ";
		
		assertThrows( SmartScriptParserException.class, () -> new SmartScriptParser(s), "The test should throw a SmartScriptParserException, but it doesn't" );  
	}
	
	
	@Test
	public void testEchoNoArguments() {
		
		String s = "{$= $} ";
		
		assertThrows( SmartScriptParserException.class, () -> new SmartScriptParser(s), "The test should throw a SmartScriptParserException, but it doesn't" );  
	}
	
	
	@Test
	public void testEmptyTag() {
		
		String s = "{$$} ";
		
		assertThrows( SmartScriptParserException.class, () -> new SmartScriptParser(s), "The test should throw a SmartScriptParserException, but it doesn't" );  
	}
	
	
	@Test
	public void testForMissingQuotationMark() {
		
		String s = "{$ FOR year \"1 10 1 3 $} ";
		
		assertThrows( SmartScriptParserException.class, () -> new SmartScriptParser(s), "The test should throw a SmartScriptParserException, but it doesn't" );  
	}
	
}
