package hr.fer.oprpp1.hw08.local;

public class LocalizationProviderBridge extends AbstractLocalizationProvider {

	private boolean connected;
	private ILocalizationListener listener;
	private ILocalizationProvider parent;
	private String currentLanguage;
	
	public LocalizationProviderBridge(ILocalizationProvider p) {
		this.parent = p;
		this.currentLanguage = p.getCurrentLanguage();
		listener = () -> this.fire();
	}

	public void disconnect() {
		if(!connected) return;
		parent.removeLocalizationListener(listener);
		connected = false;
	}
	
	public void connect() {
		if(connected) return;
		parent.addLocalizationListener(listener);
		connected = true;
	}
	
	@Override
	public String getString(String key) {
		return parent.getString(key);
	}

	@Override
	public String getCurrentLanguage() {
		return this.currentLanguage;
	}
	
}
