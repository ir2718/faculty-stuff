package hr.fer.zemris.java.hw05.shell.commands;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;

import hr.fer.zemris.java.hw05.shell.Environment;

public interface DataReader {

	default void readAndDo(Environment env, Path path, byte[] buff) {
		try (InputStream is = Files.newInputStream(path) ) {
			doBefore();
			while(true) {
				int r = is.read(buff);
				if(r<1) break;
				doWhile(r);
			}
			doAfter();
			is.close(); 
		} catch(IOException exc) {
			env.writeln(exc.getMessage());
			env.writeln("Something went wrong while reading or writing the files.");
		}
	}
	
	default void doBefore() { }
	void doWhile(int r);
	default void doAfter() { }

}
