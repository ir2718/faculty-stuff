package zad6;

public interface ISubject {
	
	void addListener(IListener l);

	void removeListener(IListener l);

	void notifyListeners();

}
