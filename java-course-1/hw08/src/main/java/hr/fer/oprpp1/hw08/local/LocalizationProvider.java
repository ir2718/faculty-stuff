package hr.fer.oprpp1.hw08.local;

import java.util.Locale;
import java.util.ResourceBundle;

public class LocalizationProvider extends AbstractLocalizationProvider{

	private String language;
	private ResourceBundle bundle;

	private static final LocalizationProvider INSTANCE = new LocalizationProvider();
	
	private LocalizationProvider() {
		this.language = "en" ;
		this.bundle = ResourceBundle.getBundle("hr.fer.oprpp1.hw08.local.prijevodi", Locale.forLanguageTag(language));
	}
	
	public static LocalizationProvider getInstance() {
		return INSTANCE;
	}

	public void setLanguage(String lang) {
		this.language = lang;
		this.bundle = ResourceBundle.getBundle("hr.fer.oprpp1.hw08.local.prijevodi", Locale.forLanguageTag(lang));
		this.fire();
	}
	
	public String getString(String key) {
		return this.bundle.getString(key);
	}

	@Override
	public String getCurrentLanguage() {
		return this.language;
	}
	
}
