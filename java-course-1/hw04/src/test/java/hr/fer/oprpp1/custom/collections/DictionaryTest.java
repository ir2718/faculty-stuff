package hr.fer.oprpp1.custom.collections;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;

public class DictionaryTest {

	@Test
	public void testEmptyDicitionary() {
		
		Dictionary<String, String> dictionary = new Dictionary<>();
		assertTrue(dictionary.isEmpty(), "The dictionary should be empty, but it's size isn't 0.");
		
	}
	
	@Test
	public void testPutAndGetFewElements() {
		
		Dictionary<String, Integer> dictionary = new Dictionary<>();
		dictionary.put("a", 65);
		dictionary.put("b", 66);
		dictionary.put("c", 67);
		
		assertEquals(dictionary.get("a"), 65, "The value of the element with the key \"a\" should be 65 but it is "+dictionary.get(0)+".");
		assertEquals(dictionary.get("b"), 66, "The value of the element with the key \"b\" should be 66 but it is "+dictionary.get(1)+".");
		assertEquals(dictionary.get("c"), 67, "The value of the element with the key \"c\" should be 67 but it is "+dictionary.get(2)+".");
	}
	
	
	@Test
	public void testRemoveFewElements() {
		
		Dictionary<Integer, Integer> dictionary = new Dictionary<>();
		dictionary.put(1, 6);
		dictionary.put(2, 28);
		dictionary.put(3, 496);
		dictionary.put(4, 8128);
		dictionary.remove(4);
		
		assertEquals( null, dictionary.get(4), "The returned value should be null but it is "+dictionary.get(4)+"." );
	}
	
	@Test
	public void testRemoveEmpty() {
		
		Dictionary<Integer, Integer> dictionary = new Dictionary<>();
		Integer value = dictionary.remove(4);
		
		assertEquals( null, value, "The returned value should be null but it is "+value+"." );
	}
	
	@Test
	public void testPutWithExistingKey() {
		
		Dictionary<String, Integer> dictionary = new Dictionary<>();
		dictionary.put( "e", 3);
		Integer value = dictionary.put("e", 2);
		
		assertEquals( 3, value, "The returned value should be 3 but it is "+value+"." );
		assertEquals( 2, dictionary.get("e"), "The returned value should be 2 but it is "+dictionary.get("e")+"." );
	}
	
	@Test
	public void testTenElements() {
		
		Dictionary<String, Integer> dictionary = new Dictionary<>();
		dictionary.put("first", 1);
		dictionary.put("second", 2);
		dictionary.put("third", 3);
		dictionary.put("fourth", 4);
		dictionary.put("fifth", 5);
		dictionary.put("sixth", 6);
		dictionary.put("seventh", 7);
		dictionary.put("eigth", 8);
		dictionary.put("ninth", 9);
		dictionary.put("tenth", 10);
		
		assertEquals( 1, dictionary.get("first"), "The returned value should be 1 but it is "+dictionary.get("first")+"." );
		assertEquals( 2, dictionary.get("second"), "The returned value should be 2 but it is "+dictionary.get("second")+"." );
		assertEquals( 3, dictionary.get("third"), "The returned value should be 3 but it is "+dictionary.get("third")+"." );
		assertEquals( 4, dictionary.get("fourth"), "The returned value should be 4 but it is "+dictionary.get("fourth")+"." );
		assertEquals( 5, dictionary.get("fifth"), "The returned value should be 5 but it is "+dictionary.get("fifth")+"." );
		assertEquals( 6, dictionary.get("sixth"), "The returned value should be 6 but it is "+dictionary.get("sixth")+"." );
		assertEquals( 7, dictionary.get("seventh"), "The returned value should be 7 but it is "+dictionary.get("seventh")+"." );
		assertEquals( 8, dictionary.get("eigth"), "The returned value should be 8 but it is "+dictionary.get("eigth")+"." );
		assertEquals( 9, dictionary.get("ninth"), "The returned value should be 9 but it is "+dictionary.get("ninth")+"." );
		assertEquals( 10, dictionary.get("tenth"), "The returned value should be 10 but it is "+dictionary.get("tenth")+"." );
		

	}
	
}
