package zad5;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.List;
import java.util.stream.Collectors;

public class WriteListener implements IListener {
	
	private SlijedBrojeva subject;
	private Path p;
	
	public WriteListener(SlijedBrojeva subject, Path p) {
		this.subject = subject;
		this.p = p;
	}
	
	@Override
	public void update() {
		List<Integer> l = subject.getCollection();
		String s = l.stream()
				.map(e -> e.toString())
				.collect(Collectors.joining(","));
		DateTimeFormatter dtf = DateTimeFormatter.ofPattern("yyyy/MM/dd HH:mm:ss");  
		LocalDateTime now = LocalDateTime.now();  
		s+= "\n" + dtf.format(now);
		
		try {
			OutputStream os = Files.newOutputStream(p);
			os.write(s.getBytes());
			os.close();
		} catch (IOException e) { }
		
	}

}
