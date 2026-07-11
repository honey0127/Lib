package com.example.rag_library

import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.IOException

/**
 * [RagEngine] 스냅샷의 메타(청크 텍스트 + 임베더 차원) 직렬화.
 * 순수 JVM 코드 — 호스트 유닛테스트로 검증한다.
 *
 * 형식(v2): writeUTF(MAGIC) | int dim | int chunkCount | (int id, docId, index, text)*
 * v2 부터 청크별 id 를 함께 저장한다 — removeDocument 이후 id 가 연속이 아닐 수 있어서다.
 *
 * 주의: DataOutputStream.writeUTF 는 문자열당 64KB(UTF-8 바이트) 제한이 있다 —
 * 기본 청킹(chunkSize 500 + overlap 100)에서는 여유가 크다.
 */
internal object ChunkStore {

    private const val MAGIC = "RAG-META-v2"

    fun save(out: DataOutputStream, dim: Int, chunks: Map<Int, Chunk>) {
        out.writeUTF(MAGIC)
        out.writeInt(dim)
        out.writeInt(chunks.size)
        for ((id, c) in chunks) {
            out.writeInt(id)
            out.writeUTF(c.docId)
            out.writeInt(c.index)
            out.writeUTF(c.text)
        }
    }

    /** @return (임베더 차원, id→청크 맵) — 형식이 다르거나 손상이면 [IOException] */
    fun load(input: DataInputStream): Pair<Int, LinkedHashMap<Int, Chunk>> {
        val magic = input.readUTF()
        if (magic != MAGIC) {
            throw IOException("RagEngine 메타 파일이 아니거나 구버전임 (magic=$magic)")
        }
        val dim = input.readInt()
        val count = input.readInt()
        if (dim <= 0 || count < 0) {
            throw IOException("메타 파일 손상 (dim=$dim, count=$count)")
        }
        val chunks = LinkedHashMap<Int, Chunk>(count * 2)
        repeat(count) {
            val id = input.readInt()
            if (id < 0) throw IOException("메타 파일 손상 (id=$id)")
            val docId = input.readUTF()
            val index = input.readInt()
            val text = input.readUTF()
            if (chunks.put(id, Chunk(docId, index, text)) != null) {
                throw IOException("메타 파일 손상 (중복 id=$id)")
            }
        }
        return dim to chunks
    }
}
