package hr.fer.oprpp1.hw08.local;

import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;

import javax.swing.JFrame;

public class FormLocalizationProvider extends LocalizationProviderBridge{

	public FormLocalizationProvider(ILocalizationProvider p, JFrame f) {
		super(p);
		
		f.addWindowListener(new WindowAdapter() {

			@Override
			public void windowOpened(WindowEvent e) { connect(); }

			@Override
			public void windowClosed(WindowEvent e) { disconnect(); }

		});
	}
	
}
