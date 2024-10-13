package hr.fer.oprpp1.custom.collections;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

public class LinkedListIndexedCollectionTest {
	
	@Test
	public void testFirstConstructor() {
		assertEquals( 0, new LinkedListIndexedCollection().size(), "The size of test LinkedListIndexedCollection isn't 0!" );
	}

	@Test
	public void testSecondConstructor() {
		LinkedListIndexedCollection c = new LinkedListIndexedCollection();
		c.add(2); c.add(4); c.add(6);
		LinkedListIndexedCollection c2 = new LinkedListIndexedCollection(c);
		assertEquals( 3, c2.size(), "The size of test LinkedListIndexedCollection isn't 3! It is "+c2.size()+".");
	}
	
	@Test
	public void testAddThreeObjectsToLinkedList() {
		LinkedListIndexedCollection c = new LinkedListIndexedCollection();
		c.add(2); c.add("abcd"); c.add(true);;
		assertEquals( 3, c.size(), "The size of test LinkedListIndexedCollection isn't 3! It is "+c.size()+".");
	}
	
	@Test
	public void testGetObjectAtIndex() {
		LinkedListIndexedCollection c = new LinkedListIndexedCollection();
		c.add(2); c.add("abcd"); c.add(true);c.add(123);
		assertEquals( 123, c.get(3), "The value at index 3 should be 123! It is "+c.get(3)+".");
		assertEquals( true, c.get(2), "The value at index 2 should be true! It is "+c.get(2)+".");
		assertEquals( "abcd", c.get(1), "The value at index 1 should be abcd! It is "+c.get(1)+".");
		assertEquals( 2, c.get(0), "The value at index 0 should be 2! It is "+c.get(0)+".");
		assertThrows( IndexOutOfBoundsException.class, () -> c.get(-1) , "C.get(-1) should throw IndexOutOfBoundsException!");
	}
	
	@Test
	public void testClearMethod() {
		LinkedListIndexedCollection c = new LinkedListIndexedCollection();
		c.add(2); c.add("abcd"); c.add(true);c.add(123);
		c.clear();
		assertEquals(null, c.getFirst() , "C.getFirst() should be null but it isn't! It is "+c.getFirst()+".");
		assertEquals(null, c.getLast() , "C.getLastt() should be null but it isn't! It is "+c.getLast()+".");
		assertThrows(IndexOutOfBoundsException.class, () -> c.get(1), "The list should be empty and throw IndexOutOfBoundsException!");
	}
	
	@Test
	public void testInsertThreeValuesIntoLinkedList() {
		LinkedListIndexedCollection c = new LinkedListIndexedCollection();
		c.insert(123, 0); c.insert(122, 1); c.insert(121, 2);
		c.insert(120, 0); c.insert(123123, c.size());
		assertEquals(120, c.get(0) , "120 should be the first element but it isn't! It is "+c.get(0)+".");
		assertEquals(123123, c.get(c.size()-1) , "123123 should be the last element but it isn't! It is "+c.get(c.size()-1)+".");
		assertThrows(IndexOutOfBoundsException.class, () -> c.get(-1), "The method should throw IndexOutOfBoundsException!");
	}

	@Test
	public void testShouldRemoveThreeValuesInTheLinkedList() {
		LinkedListIndexedCollection c = new LinkedListIndexedCollection();
		c.insert(123, 0); c.insert(122, 1); c.insert(121, 2);
		c.insert(120, 0); c.insert(123123, c.size());
		c.remove(0);
		assertEquals(123, c.get(0) , "123 should be the first element but it isn't! It is "+c.get(0)+".");
		c.remove(c.size()-1);
		assertEquals(121, c.get(c.size()-1) , "121 should be the last element but it isn't! It is "+c.get(c.size()-1)+".");
		assertThrows(IndexOutOfBoundsException.class, () -> c.remove(-1), "The method should throw IndexOutOfBoundsException!");
	}
	
	
	
	

	
	
	@Test
	void testConstructors() {
		LinkedListIndexedCollection col1 = new LinkedListIndexedCollection();
		assertEquals(0, col1.size());
		assertThrows(IndexOutOfBoundsException.class, () -> col1.get(0));
		for (int i = 1; i < 6; i++) {
			col1.add(i);
		}
		
		LinkedListIndexedCollection col2 = new LinkedListIndexedCollection();
		col2.addAll(col1);
		//System.out.println(col1.size());
		assertEquals(col1.size(), col2.size());
		
	}
	
	@Test
	void testToArray() {
		LinkedListIndexedCollection col = new LinkedListIndexedCollection();
		Object[] array = {1,2,3,4,5};
		for (int i = 1; i < 6; i++) {
			col.add(i);
		}
		assertArrayEquals(array, col.toArray());
	}
	
	@Test
	void testAdd() {
		LinkedListIndexedCollection col = new LinkedListIndexedCollection();
		int number = 5;
		col.add(5);
		col.add("test str");
		assertThrows(NullPointerException.class, () -> col.add(null));
		assertEquals(number, col.get(0));
		assertEquals("test str", col.get(1));
	}
	
	@Test
	void testGet() {
		LinkedListIndexedCollection col = new LinkedListIndexedCollection();
		col.add(5);
		assertThrows(IndexOutOfBoundsException.class, () -> col.get(-1));
		assertThrows(IndexOutOfBoundsException.class, () -> col.get(2));
	}
	
	@Test
	void testContains() {
		LinkedListIndexedCollection col = new LinkedListIndexedCollection();
		col.add(5);
		col.add(2);
		col.add("value");
		assertTrue(col.contains("value"));
		assertFalse(col.contains("value2"));	
	}
	
	@Test
	void testClear() {
		LinkedListIndexedCollection col = new LinkedListIndexedCollection();
		col.add(13);
		col.add(14);
		col.clear();
		assertEquals(0, col.size());
		
	}
	
	@Test
	void testInsert() {
		LinkedListIndexedCollection col = new LinkedListIndexedCollection();
		for (int i = 1; i < 6; i++) {
			col.add(i);
		}
		System.out.println("Size: " + col.size());
		assertThrows(IndexOutOfBoundsException.class, () -> col.insert("value", -1));
		assertThrows(IndexOutOfBoundsException.class, () -> col.insert("value", 6));
		col.insert("value", 2);
		col.insert(7, 4);
		assertEquals("value", col.get(2));
		assertEquals(7, col.get(4));
		assertEquals(7, col.size());
	}
	
	@Test
	void testRemove() {
		LinkedListIndexedCollection col = new LinkedListIndexedCollection();
		for (int i = 1; i < 6; i++) {
			col.add(i);
		}
		System.out.println(col.size());
		col.remove(4);
		assertEquals(4, col.size());
		assertFalse(col.contains(7));
	}
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	@Test
	public void testConstructor() {
		LinkedListIndexedCollection list = new LinkedListIndexedCollection();
		assertEquals(0, list.size());
		list.add("Hello world");
		list.add(1919);
		ArrayIndexedCollection col = new ArrayIndexedCollection(3);
		col.add(Integer.valueOf(20));
		col.add("New York");
		col.add("San Francisco");
		LinkedListIndexedCollection col2 = new LinkedListIndexedCollection(col);
		assertArrayEquals(col.toArray(), col2.toArray());
		LinkedListIndexedCollection list2 = new LinkedListIndexedCollection(list);
		assertArrayEquals(list.toArray(), list2.toArray());
		assertThrows(NullPointerException.class,()->new LinkedListIndexedCollection((LinkedListIndexedCollection)null));
		assertThrows(NullPointerException.class,()->new LinkedListIndexedCollection((ArrayIndexedCollection)null));
	}
	@Test
	public void testAdd2() {
		LinkedListIndexedCollection list = new LinkedListIndexedCollection();
		list.add("Hello");
		assertEquals("Hello", list.toArray()[0]);
		assertThrows(NullPointerException.class, ()->list.add(null));
	}
	@Test
	public void testGet2() {
		LinkedListIndexedCollection list = new LinkedListIndexedCollection();
		list.add("Hello");
		assertEquals("Hello",list.get(0));
		assertThrows(IndexOutOfBoundsException.class, ()->list.get(15));
	}
	@Test
	public void testClear2() {
		LinkedListIndexedCollection list = new LinkedListIndexedCollection();
		for(int i=0;i<10;i++)list.add(i);
		list.clear();
		assertArrayEquals(new Object[0], list.toArray());
	}
	@Test
	public void testInsert2() {
		LinkedListIndexedCollection list = new LinkedListIndexedCollection();
		for(int i=0;i<5;i++)list.add(i);
		list.insert(15,3);
		assertArrayEquals(new Object[] {0,1,2,15,3,4}, list.toArray());
		assertThrows(IndexOutOfBoundsException.class, ()->list.insert(15, 7));
		assertThrows(NullPointerException.class,()->list.insert(null, 1));
	}
	@Test
	public void testIndexOf() {
		LinkedListIndexedCollection list = new LinkedListIndexedCollection();
		for(int i=0;i<5;i++)list.add(i);
		assertEquals(3, list.indexOf(3));
		assertEquals(-1, list.indexOf(196));
		assertEquals(-1, list.indexOf(null));
	}
	@Test
	public void testRemove2() {
		LinkedListIndexedCollection list = new LinkedListIndexedCollection();
		for(int i=0;i<5;i++)list.add(i);
		list.remove(1);
		assertEquals(-1, list.indexOf(1));
		assertEquals(0, list.indexOf(0));
		assertEquals(1, list.indexOf(2));
	}
	
}
