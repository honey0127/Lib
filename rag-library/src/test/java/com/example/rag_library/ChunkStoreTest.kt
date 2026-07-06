package com.example.rag_library

import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.IOException
import org.junit.Assert.assertEquals
import org.junit.Test

class ChunkStoreTest {

    private fun roundTrip(dim: Int, chunks: List<Chunk>): Pair<Int, List<Chunk>> {
        val bos = ByteArrayOutputStream()
        DataOutputStream(bos).use { ChunkStore.save(it, dim, chunks) }
        return DataInputStream(ByteArrayInputStream(bos.toByteArray())).use { ChunkStore.load(it) }
    }

    @Test
    fun roundTrip_preservesChunksAndDim() {
        val chunks = listOf(
            Chunk("doc1", 0, "김치는 발효 음식이다.\n줄바꿈 포함 텍스트"),
            Chunk("doc1", 1, "탭\t문자와 이모지 🙂 포함"),
            Chunk("doc-2", 0, "second document"),
        )
        val (dim, loaded) = roundTrip(256, chunks)
        assertEquals(256, dim)
        assertEquals(chunks, loaded)
    }

    @Test
    fun emptyChunkList_roundTrips() {
        val (dim, loaded) = roundTrip(384, emptyList())
        assertEquals(384, dim)
        assertEquals(emptyList<Chunk>(), loaded)
    }

    @Test(expected = IOException::class)
    fun badMagic_throwsIOException() {
        val bos = ByteArrayOutputStream()
        DataOutputStream(bos).use {
            it.writeUTF("NOT-RAG-META")
            it.writeInt(256)
            it.writeInt(0)
        }
        DataInputStream(ByteArrayInputStream(bos.toByteArray())).use { ChunkStore.load(it) }
    }
}
