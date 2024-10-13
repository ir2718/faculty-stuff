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

}
