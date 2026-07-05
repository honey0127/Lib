package com.example.rag_library

import java.io.BufferedReader
import java.io.File
import java.text.Normalizer

/**
 * SentencePiece Unigram 토크나이저의 순수 Kotlin 구현 (네이티브 의존성 없음).
 *
 * multilingual-e5-small(XLM-RoBERTa 계열)의 HuggingFace `tokenizer.json`을
 * `scripts/prepare_model.py` 로 변환한 vocab TSV 를 읽어 동작한다.
 *
 * TSV 형식:
 * ```
 * #sp-unigram-v1<TAB>unk=3<TAB>bos=0<TAB>eos=2
 * <piece><TAB><score>      ← 줄 번호(0부터) == 토큰 id
 * ...
 * ```
 *
 * 알고리즘: Viterbi 최적 분할 — 각 위치에서 vocab 조각 매칭으로 전이하며
 * 로그확률(score) 합이 최대인 분할을 찾는다. 어떤 조각도 못 찾는 문자는
 * `<unk>` 로 전이하고, 출력 시 연속 `<unk>` 는 하나로 병합한다(SentencePiece 규칙).
 *
 * 정규화: NFKC + 공백 정리 + metaspace(공백→'▁', 문두 '▁' 부착).
 * (원본 XLM-R 은 precompiled charsmap 을 쓰지만 NFKC 가 실용적 근사다 — 한글/영문에선 사실상 동일)
 */
class UnigramTokenizer private constructor(
    private val vocab: HashMap<String, Int>,
    private val scores: FloatArray,
    private val unkId: Int,
    private val bosId: Int,
    private val eosId: Int,
    private val maxPieceLen: Int,
    private val unkPenalty: Double,
) : Tokenizer {

    val vocabSize: Int get() = scores.size

    override fun encode(text: String, maxTokens: Int): IntArray {
        require(maxTokens >= 2) { "maxTokens must be >= 2 (bos/eos 포함)" }
        val s = normalize(text)
        if (s.isEmpty()) return intArrayOf(bosId, eosId)

        // Viterbi: bestScore[i] = s[0..i) 최적 분할 점수, bestLen[i] = 그 경로의 마지막 조각 길이(0=unk 1글자)
        val n = s.length
        val bestScore = DoubleArray(n + 1) { Double.NEGATIVE_INFINITY }
        val bestLen = IntArray(n + 1)
        bestScore[0] = 0.0
        for (i in 0 until n) {
            if (bestScore[i] == Double.NEGATIVE_INFINITY) continue
            val maxL = minOf(maxPieceLen, n - i)
            for (len in 1..maxL) {
                val id = vocab[s.substring(i, i + len)] ?: continue
                val cand = bestScore[i] + scores[id]
                if (cand > bestScore[i + len]) {
                    bestScore[i + len] = cand
                    bestLen[i + len] = len
                }
            }
            // vocab 에 없는 문자를 위한 <unk> 한 글자 전이 (항상 도달 가능성 보장)
            val cand = bestScore[i] + unkPenalty
            if (cand > bestScore[i + 1]) {
                bestScore[i + 1] = cand
                bestLen[i + 1] = 0
            }
        }

        // 역추적 (뒤→앞) 후 뒤집으면서 연속 unk 병합
        val reversed = ArrayList<Int>()
        var pos = n
        while (pos > 0) {
            val len = bestLen[pos]
            if (len == 0) {
                reversed.add(unkId)
                pos -= 1
            } else {
                reversed.add(vocab.getValue(s.substring(pos - len, pos)))
                pos -= len
            }
        }
        val body = ArrayList<Int>(reversed.size)
        for (i in reversed.indices.reversed()) {
            val id = reversed[i]
            if (id == unkId && body.isNotEmpty() && body.last() == unkId) continue
            body.add(id)
        }

        val bodyLen = minOf(body.size, maxTokens - 2)
        val out = IntArray(bodyLen + 2)
        out[0] = bosId
        for (i in 0 until bodyLen) out[i + 1] = body[i]
        out[bodyLen + 1] = eosId
        return out
    }

    private fun normalize(text: String): String {
        val nfkc = Normalizer.normalize(text, Normalizer.Form.NFKC)
        val collapsed = nfkc.trim().replace(WHITESPACE, " ")
        if (collapsed.isEmpty()) return ""
        return METASPACE + collapsed.replace(' ', METASPACE)
    }

    companion object {
        private const val METASPACE = '▁' // U+2581
        private val WHITESPACE = Regex("\\s+")
        private const val HEADER_PREFIX = "#sp-unigram-v1"
        private const val MAX_PIECE_LEN_CAP = 64

        fun load(file: File): UnigramTokenizer = file.bufferedReader().use { load(it) }

        fun load(reader: BufferedReader): UnigramTokenizer {
            val header = reader.readLine()
                ?: throw IllegalArgumentException("빈 vocab 파일")
            require(header.startsWith(HEADER_PREFIX)) {
                "vocab 헤더가 '$HEADER_PREFIX' 로 시작해야 함 (scripts/prepare_model.py 로 생성): $header"
            }
            val meta = header.split('\t').drop(1).associate {
                val (k, v) = it.split('=', limit = 2)
                k to v.toInt()
            }
            val unk = requireNotNull(meta["unk"]) { "헤더에 unk= 누락" }
            val bos = requireNotNull(meta["bos"]) { "헤더에 bos= 누락" }
            val eos = requireNotNull(meta["eos"]) { "헤더에 eos= 누락" }

            val vocab = HashMap<String, Int>()
            val scoreList = ArrayList<Float>()
            var maxLen = 1
            var minScore = 0.0f
            reader.forEachLine { line ->
                if (line.isEmpty()) return@forEachLine
                val tab = line.lastIndexOf('\t')
                require(tab > 0) { "잘못된 vocab 줄: $line" }
                val piece = line.substring(0, tab)
                val score = line.substring(tab + 1).toFloat()
                val id = scoreList.size
                scoreList.add(score)
                // 동일 piece 중복 시 첫 항목 우선 (HF vocab 은 중복 없음)
                vocab.putIfAbsent(piece, id)
                if (piece.length > maxLen) maxLen = piece.length
                if (score < minScore) minScore = score
            }
            require(scoreList.isNotEmpty()) { "vocab 이 비어 있음" }
            require(unk in scoreList.indices && bos in scoreList.indices && eos in scoreList.indices) {
                "unk/bos/eos id 가 vocab 범위를 벗어남"
            }
            return UnigramTokenizer(
                vocab = vocab,
                scores = scoreList.toFloatArray(),
                unkId = unk,
                bosId = bos,
                eosId = eos,
                maxPieceLen = minOf(maxLen, MAX_PIECE_LEN_CAP),
                unkPenalty = minScore.toDouble() - 10.0,
            )
        }
    }
}
