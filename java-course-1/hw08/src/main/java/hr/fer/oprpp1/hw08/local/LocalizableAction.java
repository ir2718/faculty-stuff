package hr.fer.oprpp1.hw08.local;

import java.awt.event.ActionEvent;

import javax.swing.AbstractAction;

public abstract class LocalizableAction extends AbstractAction{

	private String key;
	private ILocalizationListener listener;
	private ILocalizationProvider prov;
	
	public LocalizableAction(String key,ILocalizationProvider provider) {
		this.key = key;
		this.prov = provider;
		this.listener = () -> update();
		provider.addLocalizationListener(this.listener);
		update();
	}

	private void update() {
		this.putValue(NAME, this.prov.getString(key));
		this.putValue(SHORT_DESCRIPTION, this.prov.getString(key+"Desc"));
	}


}
