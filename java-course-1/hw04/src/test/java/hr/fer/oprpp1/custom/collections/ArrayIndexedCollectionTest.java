package hr.fer.oprpp1.custom.collections;

import static org.junit.jupiter.api.Assertions.*;

import java.util.Objects;

import org.junit.jupiter.api.Test;

public class ArrayIndexedCollectionTest {

	private static final int SIZE = 16;

	@Test
	public void testFirstConstructor() {
		assertEquals(SIZE, new ArrayIndexedCollection().getElementsLength(), "The constructor doesn't allocate room for 16 elements!");
	}

	@Test
	public void testSecondConstructorShouldThrowIllegalArgExc() {
		assertThrows( IllegalArgumentException.class, () -> new ArrayIndexedCollection(0) );
	} 

	@Test
	public void testSecondConstructorShouldAllocateMemoryForTenElements() {
		assertEquals( 10, new ArrayIndexedCollection(10).getElementsLength(), "The constructor didn't allocate memory for 10 elements!" );
	} 

	@Test
	public void testThirdConstructorShouldAllocateMemoryForTwentyElements() {
		ArrayIndexedCollection c = new ArrayIndexedCollection(20);

		for(int i=0; i<20; i++)
			c.add(i);

		assertEquals( 20, c.getElementsLength(), "The constructor didn't allocate memory for twenty elements!" );
	}

	@Test
	public void testThirdConstructorSizeShouldBeNineteen() {
		ArrayIndexedCollection c = new ArrayIndexedCollection(20);
		for(int i=0; i<19; i++)
			c.add(i);
		ArrayIndexedCollection c2 = new ArrayIndexedCollection(c);

		assertEquals( 19, c2.getElementsLength(), "The constructor didn't allocate memory for twenty elements! "
				+"It allocated memory for "+c2.getElementsLength()+" objects." );
	}

	@Test
	public void testThirdConstructorShouldThrowNullPtrExc() {
		assertThrows( NullPointerException.class, () -> new ArrayIndexedCollection(null) );
	}

	@Test
	public void testFourthConstructorShouldThrowNullPtrExc() {
		assertThrows( NullPointerException.class, () -> new ArrayIndexedCollection(null, 5) );
	}

	@Test
	public void testMethodAddShouldThrowNullPtrException() {
		assertThrows( NullPointerException.class, () -> new ArrayIndexedCollection().add(null) );
	}

	@Test
	public void testMethodAddShouldAddFirstSixteenNumbersIntoElementsArray() {
		ArrayIndexedCollection c = new ArrayIndexedCollection();
		Object[] expected = new Object[16];
		
		for(int i=0; i<SIZE; i++) {
			c.add(i); 
			expected[i] = i;
		}
		
		assertArrayEquals( expected, c.toArray() );
	}
	
	@Test
	public void testMethodGetShouldThrowIndexOutOfBoundsExc() {
		assertThrows( IndexOutOfBoundsException.class, () -> new ArrayIndexedCollection().get(0) );
	}
	
	@Test
	public void testMethodGetShouldGetValueAtIndexThreeAndClearItAfter() {
		ArrayIndexedCollection c = new ArrayIndexedCollection();
		for(int i=0; i<SIZE; i++)
			c.add(i);
		assertEquals(3, c.get(3), "The method didn't get the element at the index 3!");
		
		c.clear();
		Object[] arr = new Object[16];
		assertArrayEquals( arr, c.toArray() );
	}
	
	@Test
	public void testInsertNumbersIntoArrayAndCheckWhereTheyAre() {
		ArrayIndexedCollection c = new ArrayIndexedCollection();
		for(int i=0; i<SIZE; i++)
			c.add(i);
		c.insert(50, c.size());
		c.insert(100, 0);
		c.insert(77, 4);
		assertEquals(c.indexOf(50), c.size()-1, "The inserted number 50 isn't at the end of the array!");
		assertEquals(c.indexOf(100), 0, "The inserted number 100 isn't at the beginning  of the array!");
		assertEquals(c.indexOf(77), 4, "The inserted number 77 isn't at the 4th position of the array!");
	}
	
	@Test
	public void testIndexOfMethod() {
		ArrayIndexedCollection c = new ArrayIndexedCollection();
		c.add(2); c.add(3); c.add(0); c.add(66);
		assertEquals(0, c.indexOf(2), "Index of the value 2 isn't 0 but it should be!");
		assertEquals(2,c.indexOf(0), "Index of the value 0 isn't 2 but it should be!");
		assertEquals(-1, c.indexOf(444), "The value 444 is in the array but it shouldn't be!");
	}
	
	@Test
	public void testRemoveMethodThrowsIndexOutOfBoundsExc() {
		ArrayIndexedCollection c = new ArrayIndexedCollection();
		c.add(2); c.add(21); c.add(321); c.add(111);
		c.remove(0);
		assertEquals(21, c.get(0));
		assertThrows( IndexOutOfBoundsException.class, () -> c.remove(c.size()) );
	}
	
}
