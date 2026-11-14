package org.safora.ai
import org.junit.Assert.*
import org.junit.Test
class AudioRingTest {
    @Test fun arbitraryBlocksPreserveWindowAndTimestamp() {
        val source=ShortArray(40000) { (it%32000).toShort() }
        val ring=AudioRing();val times=mutableListOf<Long>();var index=0
        for(start in source.indices step 317) {
            val block=source.copyOfRange(start,minOf(start+317,source.size))
            ring.add(block,block.size) { window,time ->
                assertEquals(source[index*8000]/32768f,window[0],0f)
                assertEquals(source[index*8000+15599]/32768f,window.last(),0f)
                times.add(time);index++
            }
        }
        assertEquals(listOf(975L,1475L,1975L,2475L),times)
        ring.clear()
    }
}
