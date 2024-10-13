package hr.fer.oprpp1.custom.scripting.parser;

import hr.fer.oprpp1.custom.collections.ArrayIndexedCollection;

import hr.fer.oprpp1.custom.collections.ObjectStack;
import hr.fer.oprpp1.custom.scripting.elems.*;
import hr.fer.oprpp1.custom.scripting.lexer.*;
import hr.fer.oprpp1.custom.scripting.nodes.*;

/**
 * The class is used for parsing the elements from the lexer
 * into different nodes.
 * 
 * @author Ivan Rep
 */
public class SmartScriptParser {

	private ObjectStack stack;
	private SmartScriptLexer lexer;
	
	/**
	 * The constructor method for the parser.
	 * 
	 * @param documentBody the string that represents the document
	 */
	public SmartScriptParser(String documentBody) {
		lexer = new SmartScriptLexer(documentBody);
		stack = new ObjectStack();
		parse(documentBody);
	}


	/**
	 * The getter method for the document node after building
	 * the generative tree.
	 * 
	 * @return an object of type DocumentNode
	 */
	public DocumentNode getDocumentNode() {
		try {
			return (DocumentNode) stack.peek();
		}catch(ClassCastException exc) {
			throw new SmartScriptParserException("One or more tag isn't closed");
		}
	}

	
	/**
	 * The method is used for parsing the elements 
	 * into three types of nodes: EchoNode, TextNode or ForLoopNode.
	 * 
	 * @param documentBody the string representation of the document
	 */
	private void parse(String documentBody) {
		DocumentNode docNode = new DocumentNode(new ArrayIndexedCollection());
		stack.push(docNode);

		Element tempElem = null;
		Node node = null;
		while( lexer.getCurrentIndex() < lexer.getDocumentLength() ) {

			try {
				tempElem = lexer.nextToken();
			} catch ( SmartScriptLexerException exc ) {
				throw new SmartScriptParserException( exc );
			}

			if( tempElem.asText().equals("FOR") || tempElem.asText().equals("ECHO") 
					|| tempElem.asText().equals("$}") || tempElem.asText().equals("END") ) {
				parseNextElement(tempElem);
			}else if( tempElem instanceof ElementString ) {
				node = new TextNode(new ArrayIndexedCollection(), tempElem.asText());
				((Node)stack.peek()).addChild(node);
			}


		}

	}



	/**
	 * The method is used for parsing different kinds
	 * of tags in the given document. 
	 * 
	 * @param tempElem the first element inside the tag
	 */
	private void parseNextElement(Element tempElem) {

		ArrayIndexedCollection arr = new ArrayIndexedCollection();

		switch( tempElem.asText()) {
		case "FOR" -> {
			int i=0;
			while( true ) {
				Element el = lexer.nextToken();
				if( el.asText().equals("$}") ) break;
				arr.add( el );
				i++;
			}

			if( !( i>=3 && i<=4 ) || !( arr.get(0) instanceof ElementVariable && checkCondition( (Element) arr.get(1) ) && checkCondition( (Element) arr.get(2) ) ) )  
				throw new SmartScriptParserException("The types of the elements in a tag don't match.");

			if( i==4 ) {
				if( !checkCondition(  (Element) arr.get(3) ) ) 
					throw new SmartScriptParserException("The types of the elements in a tag don't match.");

				ForLoopNode node = new ForLoopNode( new ArrayIndexedCollection(), (ElementVariable)arr.get(0), (Element)arr.get(1), (Element)arr.get(2), (Element)arr.get(3) );
				((Node)stack.peek()).addChild(node);
				stack.push(node);
				return;
			}

			ForLoopNode node = new ForLoopNode( new ArrayIndexedCollection(), (ElementVariable)arr.get(0), (Element)arr.get(1), (Element)arr.get(2));
			((Node)stack.peek()).addChild(node);
			stack.push(node);
		} 

		case "ECHO" -> {
			int i = 0;
			while( true ) {
				Element el = lexer.nextToken();
				if( el.asText().equals("$}") ) break;
				arr.add( el );
				i++;
			}
			
			if(i<1) throw new SmartScriptParserException("The echo node must containt at least one element.");

			Element[] arrElem = new Element[i];

			for(int j=0; j<i; j++) {
				arrElem[j]=(Element) arr.get(j);
			}

			EchoNode node = new EchoNode ( new ArrayIndexedCollection(), arrElem  );
			((Node)stack.peek()).addChild(node);
		}

		case "END" -> {
			if(stack.size()==0) throw new SmartScriptParserException("There's too many end tags in the input.");
			stack.pop();
		}

		} 

	}

	
	/**
	 * This method is used in the parseNextElement method.
	 * It checks the condition required for the
	 * construction of the FOR tag.
	 * 
	 * @param elem the element that is checked 
	 * @return true if the element is an instance of ElementConstantDouble, ElementConstantInteger or ElementVariable 
	 * otherwise false
	 */
	private boolean checkCondition(Element elem) {
		return ( elem instanceof ElementConstantDouble || elem instanceof ElementConstantInteger ||
				elem instanceof ElementVariable);
	}
}
