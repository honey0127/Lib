package com.example.rag_library

/** 문서를 분할한 검색 단위. [index]는 문서 내 청크 순번(0부터). */
data class Chunk(
    val docId: String,
    val index: Int,
    val text: String,
)

/** 검색 결과 1건. [score]는 코사인 유사도(-1 ~ 1, 클수록 유사). */
data class SearchResult(
    val chunk: Chunk,
    val score: Float,
)
