package hr.fer.oprpp1.hw08.local;

import javax.swing.JLabel;

public class LJLabel extends JLabel {

	private static final long serialVersionUID = 1L;
	protected String key;
	private ILocalizationListener listener;
	private ILocalizationProvider prov;
	
	public LJLabel( String key, ILocalizationProvider provider ) {
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
