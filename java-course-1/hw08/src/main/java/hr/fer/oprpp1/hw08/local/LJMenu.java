package hr.fer.oprpp1.hw08.local;

import javax.swing.JMenu;

public class LJMenu extends JMenu {

	private static final long serialVersionUID = 1L;
	
	protected String key;
	private ILocalizationListener listener;
	private ILocalizationProvider prov;
	
	public LJMenu( String key, ILocalizationProvider provider ) {
		this.key = key;
		this.prov = provider;
		listener = () -> updateLabel();
		updateLabel();
		provider.addLocalizationListener(listener);
	}
	
	private void updateLabel() {
		setText( this.prov.getString(key) );
	}
	
}
