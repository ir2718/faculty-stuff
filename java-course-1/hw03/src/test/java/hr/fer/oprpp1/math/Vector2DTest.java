package hr.fer.oprpp1.math;

import static java.lang.Math.toRadians;
import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

public class Vector2DTest {

		@Test
		public void testCreateNewVector() {
			
			Vector2D v = new Vector2D(2.4, 5.2);
			
			assertEquals( 2.4, v.getX(), "The expected value was 2.4 but the x coordinate was"+v.getX()+"." );
			assertEquals( 5.2, v.getY(), "The expected value was 5.2 but the y coordinate was"+v.getY()+"." );
		}
		
		
		@Test
		public void testAddThreeVectorsToTheFirst() {
			
			Vector2D v1 = new Vector2D(-3, 0);
			Vector2D v2 = new Vector2D(4, 22);
			Vector2D v3 = new Vector2D(1.1, 1.1);
			Vector2D v4 = new Vector2D(2.2, -30);
			
			v1.add(v2);
			v1.add(v3);
			v1.add(v4);
			
			assertEquals( true, Math.abs(4.3-v1.getX()) < 0.00001 , "The expected value was 4.3 but the x coordinate was"+v1.getX()+"." );
			assertEquals( true, Math.abs(-6.9-v1.getY()) < 0.00001, "The expected value was -6.9 but the y coordinate was"+v1.getY()+"." );
		}
		
		
		@Test
		public void testAddedThreeVectorsToTheFirst() {
			
			Vector2D v1 = new Vector2D(-3, 0);
			Vector2D v2 = new Vector2D(4, 22);
			Vector2D v3 = new Vector2D(1.1, 1.1);
			Vector2D v4 = new Vector2D(2.2, -30);
			
			Vector2D res = v1.added(v2).added(v3).added(v4);
		
			assertEquals( true, Math.abs(4.3-res.getX()) < 0.00001 , "The expected value was 4.3 but the x coordinate was"+res.getX()+"." );
			assertEquals( true, Math.abs(-6.9-res.getY()) < 0.00001, "The expected value was -6.9 but the y coordinate was"+res.getY()+"." );
		}
		
		
		@Test
		public void testRotateVectorBy30() {
			
			Vector2D v1 = new Vector2D(-3, 2.42);
			v1.rotate(toRadians(30));
			
			assertEquals( true, Math.abs(-3.80807-v1.getX()) < 0.0001 , "The expected value was -3.80807 but the x coordinate was"+v1.getX()+"." );
			assertEquals( true, Math.abs(0.59578-v1.getY()) < 0.0001, "The expected value was 0.59578 but the y coordinate was"+v1.getY()+"." );
		}
		
		
		@Test
		public void testRotatedVectorBy30() {
			
			Vector2D v1 = new Vector2D(-3, 2.42);
			Vector2D res = v1.rotated(toRadians(30));
			
			assertEquals( true, Math.abs(-3.80807-res.getX()) < 0.0001 , "The expected value was -3.80807 but the x coordinate was"+res.getX()+"." );
			assertEquals( true, Math.abs(0.59578-res.getY()) < 0.0001, "The expected value was 0.59578 but the y coordinate was"+res.getY()+"." );
		}
		
		
		@Test
		public void testScaleByDouble() {
			
			Vector2D v1 = new Vector2D(3.21, -1.23);
			v1.scale(2.13);
			
			assertEquals( true, Math.abs(6.8373-v1.getX()) < 0.0001 , "The expected value was 6.8373 but the x coordinate was"+v1.getX()+"." );
			assertEquals( true, Math.abs(-2.6199-v1.getY()) < 0.0001, "The expected value was -2.6199 but the y coordinate was"+v1.getY()+"." );
		}
		
		
		@Test
		public void testScaledByDouble() {
			
			Vector2D v1 = new Vector2D(3.21, -1.23);
			Vector2D res = v1.scaled(2.13);
			
			assertEquals( true, Math.abs(6.8373-res.getX()) < 0.0001 , "The expected value was 6.8373 but the x coordinate was"+res.getX()+"." );
			assertEquals( true, Math.abs(-2.6199-res.getY()) < 0.0001, "The expected value was -2.6199 but the y coordinate was"+res.getY()+"." );
		}
		
		@Test
		public void testCopyTwoVectors() {
			
			Vector2D v1 = new Vector2D(1, 5);
			Vector2D v2 = new Vector2D(-1, -5);
			
			assertEquals( 1, v1.copy().getX(), "The expected value was 1 but the x coordinate was"+v1.getX()+"." );
			assertEquals( 5, v1.copy().getY(), "The expected value was 5 but the y coordinate was"+v1.getY()+"." );
			assertEquals( -1, v2.copy().getX(), "The expected value was -1 but the x coordinate was"+v2.getX()+"." );
			assertEquals( -5, v2.copy().getY(), "The expected value was -5 but the y coordinate was"+v2.getY()+"." );
		}

		
		
		
		
		
		
		
		
		
		@Test
	    public void testConstructor() {
	        Vector2D vector = new Vector2D(0, 0);
	        //tests whether the given values are initialized properly
	        assertEquals(0, vector.getX()+vector.getY());
	    }

	    @Test
	    public void testGetters() {
	        Vector2D vector = new Vector2D(1, 2);
	        assertEquals(1, vector.getX());
	        assertEquals(2, vector.getY());
	    }

	    @Test
	    public void testAddAndAdded() {
	        Vector2D vectorOne = new Vector2D(1,2);
	        Vector2D vectorTwo = new Vector2D(3,4);
	        vectorOne.add(vectorTwo);
	        assertEquals(new Vector2D(4,6),vectorOne);
	        Vector2D vectorThree = vectorTwo.added(vectorTwo);
	        assertEquals(new Vector2D(3,4), vectorTwo);
	        assertEquals(new Vector2D(6,8), vectorThree);

	    }
	    @Test
	    public void testRotateAndRotated() {
	        Vector2D vectorOne = new Vector2D(0,1);
	        Vector2D vectorTwo = new Vector2D(1,0);
	        vectorOne.rotate(Math.PI);
	        assertEquals(new Vector2D(0,-1), vectorOne);
	        Vector2D vectorThree = vectorTwo.rotated(3*Math.PI);
	        assertEquals(new Vector2D(-1,0), vectorThree);
	        assertEquals(new Vector2D(1,0), vectorTwo);
	    }

	    @Test
	    public void testScaleAndScaled() {
	        Vector2D vectorOne = new Vector2D(1,1);
	        Vector2D vectorTwo = new Vector2D(2,2);
	        vectorOne.scale(2);
	        assertEquals(vectorTwo, vectorOne);
	        Vector2D vectorThree = vectorTwo.scaled(3);
	        assertEquals(new Vector2D(6,6), vectorThree);
	        assertEquals(new Vector2D(2,2), vectorTwo);
	    }
	    @Test
	    public void testCopy() {
	        Vector2D vectorOne = new Vector2D(1,1);
	        Vector2D vectorTwo = vectorOne.copy();
	        vectorOne.scale(15);
	        assertEquals(new Vector2D(1,1), vectorTwo);
	        assertEquals(new Vector2D(15,15), vectorOne);
	    }
}
