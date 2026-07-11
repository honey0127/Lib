package com.example.rag_library

import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.IOException
import org.junit.Assert.assertEquals
import org.junit.Test

class ChunkStoreTest {

    private fun roundTrip(dim: Int, chunks: Map<Int, Chunk>): Pair<Int, Map<Int, Chunk>> {
        val bos = ByteArrayOutputStream()
        DataOutputStream(bos).use { ChunkStore.save(it, dim, chunks) }
        return DataInputStream(ByteArrayInputStream(bos.toByteArray())).use { ChunkStore.load(it) }
    }

    @Test
    fun roundTrip_preservesChunksAndDim() {
        val chunks = linkedMapOf(
            0 to Chunk("doc1", 0, "김치는 발효 음식이다.\n줄바꿈 포함 텍스트"),
            1 to Chunk("doc1", 1, "탭\t문자와 이모지 🙂 포함"),
            2 to Chunk("doc-2", 0, "second document"),
        )
        val (dim, loaded) = roundTrip(256, chunks)
        assertEquals(256, dim)
        assertEquals(chunks, loaded)
    }

    @Test
    fun nonContiguousIds_roundTrip() {
        // removeDocument 이후 id 가 불연속인 상태를 재현
        val chunks = linkedMapOf(
            0 to Chunk("a", 0, "첫 청크"),
            5 to Chunk("b", 0, "중간 삭제 후 남은 청크"),
            9 to Chunk("c", 0, "마지막"),
        )
        val (dim, loaded) = roundTrip(384, chunks)
        assertEquals(384, dim)
        assertEquals(chunks, loaded)
    }

    @Test
    fun emptyChunkMap_roundTrips() {
        val (dim, loaded) = roundTrip(384, emptyMap())
        assertEquals(384, dim)
        assertEquals(emptyMap<Int, Chunk>(), loaded)
    }

    @Test(expected = IOException::class)
    fun oldOrWrongMagic_throwsIOException() {
        val bos = ByteArrayOutputStream()
        DataOutputStream(bos).use {
            it.writeUTF("RAG-META-v1") // 구버전 포맷은 명확히 거부
            it.writeInt(256)
            it.writeInt(0)
        }
        DataInputStream(ByteArrayInputStream(bos.toByteArray())).use { ChunkStore.load(it) }
    }
}
