package hr.fer.oprpp1.hw02;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.charset.StandardCharsets;
import java.nio.file.Paths;


import hr.fer.oprpp1.custom.scripting.nodes.DocumentNode;
import hr.fer.oprpp1.custom.scripting.parser.*;

/**
 * The class demonstrates the functionalities of a
 * SmartScriptParser object.
 * 
 * @author Ivan Rep
 *
 */
public class SmartScriptTester {

	/**
	 * The method called once the program executes
	 * 
	 * @param args the arguments of the command line, they're not used
	 */
	public static void main(String[] args) {

		for(int n = 1 ; n < 10; n++) {
			String docBody="";
			try {
				docBody = new String(
						Files.readAllBytes(Paths.get("src\\test\\resources\\extra\\primjer"+n+".txt")),
						StandardCharsets.UTF_8
						);
			} catch( IOException exc) {
				System.out.println(exc.getMessage());
			}

			SmartScriptParser parser = null;
			try {
				parser = new SmartScriptParser(docBody);
			} catch(SmartScriptParserException e) {
				System.out.println("Unable to parse document!");
				System.exit(-1);
			} catch(Exception e) {
				System.out.println("If this line ever executes, you have failed this class!");
				System.exit(-1);
			}
			DocumentNode document = parser.getDocumentNode();
			String originalDocumentBody = document.toString();
			System.out.println("primjer"+n+".txt:\n");
			System.out.println(originalDocumentBody);
			System.out.println();
			SmartScriptParser parser2 = new SmartScriptParser(originalDocumentBody);
			DocumentNode document2 = parser2.getDocumentNode();
			System.out.println(document2.toString());
			System.out.println();
			boolean same = document.equals(document2);
			System.out.println(same);
			System.out.println("----------------------");
		}

	}

}
