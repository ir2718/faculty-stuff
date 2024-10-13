package MyTextEditor;

public class LocationRange {

	private Location start;
	private Location end;
	
	public LocationRange(Location start, Location end) {
		this.start = start;
		this.end = end;
	}
	
	public Location getStart() {
		return this.start;
	}
	
	public Location getEnd() {
		return this.end;
	}
	
	public void setEnd(Location end) {
		this.end = end;
	}
	
	public void setStart(Location start) { 
		this.start = start;
	}
	
	public LocationRange sort() {
		LocationRange r = this;
		if (this.start.getY() > this.end.getY())
			r = swap();
		else if (this.start.getY() == this.end.getY() && this.start.getX() > this.end.getX())
			r = swap();
		return r;
	}
	
	
	private LocationRange swap() {
		return  new LocationRange(new Location(end), new Location(start));
	}
}
