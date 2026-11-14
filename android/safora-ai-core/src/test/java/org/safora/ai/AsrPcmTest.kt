package org.safora.ai
import org.junit.Assert.*
import org.junit.Test
class AsrPcmTest {
    @Test fun signedEndpointsAndLittleEndianMatchPython() {
        assertArrayEquals(shortArrayOf(-32768,-16384,0,16384,32767),asrPcm16(floatArrayOf(-1f,-.5f,0f,.5f,1f)))
        assertArrayEquals(byteArrayOf(0,-128,0,0,-1,127),asrPcm16LeBytes(floatArrayOf(-1f,0f,1f)))
    }
    @Test fun emptySegmentDoesNotAllocateSamples() { assertEquals(0,asrPcm16(floatArrayOf()).size) }
    @Test fun invalidSamplesAreRejected() {
        for(x in listOf(floatArrayOf(Float.NaN),floatArrayOf(Float.POSITIVE_INFINITY),floatArrayOf(1.01f),FloatArray(192001))) {
            try { asrPcm16(x);fail("Invalid input accepted") } catch(_:IllegalArgumentException) {}
        }
    }
}
