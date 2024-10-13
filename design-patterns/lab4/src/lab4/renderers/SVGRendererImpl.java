package lab4.renderers;

import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import lab4.Point;

public class SVGRendererImpl implements Renderer {

	private List<String> lines;
	private String fileName;

	public SVGRendererImpl(String fileName) {
		this.lines = new ArrayList<>();
		this.fileName = fileName;
		lines.add("<svg  xmlns=\"http://www.w3.org/2000/svg\"");
		lines.add("      xmlns:xlink=\"http://www.w3.org/2000/svg/xlink\">");
	}

	public void close() throws IOException {
		this.lines.add("</svg>");
		Files.write(Paths.get(fileName), this.lines, Charset.defaultCharset());
	}

	@Override
	public void drawLine(Point s, Point e) {
		this.lines.add("<line x1=\""+s.getX()+"\" y1=\""+s.getY()+
				"\" x2=\""+e.getX()+"\" y2=\""+e.getY()+"\" style=\"stroke:#0000ff;\"/>");
	}

	@Override
	public void fillPolygon(Point[] points) {
		String s = "<polygon points=\"";
		s += Arrays.stream(points)
				.map(p -> p.getX()+","+p.getY())
				.collect(Collectors.joining("  "));
		s += "\" style=\"stroke:#ff0000; fill:#0000ff;\"/>";
		this.lines.add(s);
	}

}
