package com.example.rag_library

import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.IOException

/**
 * [RagEngine] 스냅샷의 메타(청크 텍스트 + 임베더 차원) 직렬화.
 * 순수 JVM 코드 — 호스트 유닛테스트로 검증한다.
 *
 * 형식: writeUTF(MAGIC) | int dim | int chunkCount | (docId, index, text)*
 * 주의: DataOutputStream.writeUTF 는 문자열당 64KB(UTF-8 바이트) 제한이 있다 —
 * 기본 청킹(chunkSize 500 + overlap 100)에서는 여유가 크다.
 */
internal object ChunkStore {

    private const val MAGIC = "RAG-META-v1"

    fun save(out: DataOutputStream, dim: Int, chunks: List<Chunk>) {
        out.writeUTF(MAGIC)
        out.writeInt(dim)
        out.writeInt(chunks.size)
        for (c in chunks) {
            out.writeUTF(c.docId)
            out.writeInt(c.index)
            out.writeUTF(c.text)
        }
    }

    /** @return (임베더 차원, 청크 목록) — 형식이 다르면 [IOException] */
    fun load(input: DataInputStream): Pair<Int, List<Chunk>> {
        val magic = input.readUTF()
        if (magic != MAGIC) {
            throw IOException("RagEngine 메타 파일이 아님 (magic=$magic)")
        }
        val dim = input.readInt()
        val count = input.readInt()
        if (dim <= 0 || count < 0) {
            throw IOException("메타 파일 손상 (dim=$dim, count=$count)")
        }
        val chunks = ArrayList<Chunk>(count)
        repeat(count) {
            val docId = input.readUTF()
            val index = input.readInt()
            val text = input.readUTF()
            chunks.add(Chunk(docId, index, text))
        }
        return dim to chunks
    }
}
