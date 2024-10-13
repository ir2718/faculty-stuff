package zad5;

public interface ISubject {
	
	void addListener(IListener l);

	void removeListener(IListener l);

	void notifyListeners();
}
