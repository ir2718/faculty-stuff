package MyTextEditor;

import java.lang.reflect.Constructor;

public class PluginFactory {

	@SuppressWarnings("unchecked")
	public static Plugin getInstance(String s, String path) throws Exception {
		Class<Plugin> clazz = (Class<Plugin>) Class.forName(path + s);
		Constructor<?> ctr = clazz.getConstructor();
		Plugin p = (Plugin)ctr.newInstance();
		return p;
	}

}
