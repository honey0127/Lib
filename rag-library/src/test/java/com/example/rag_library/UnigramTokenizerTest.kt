package com.example.rag_library

import java.io.BufferedReader
import java.io.StringReader
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test

class UnigramTokenizerTest {

    // id:  0=<unk> 1=<s> 2=</s> 3=▁ 4=▁ab 5=a 6=b 7=ab 8=▁a
    private val vocabTsv = """
        #sp-unigram-v1${"\t"}unk=0${"\t"}bos=1${"\t"}eos=2
        <unk>${"\t"}0.0
        <s>${"\t"}0.0
        </s>${"\t"}0.0
        ▁${"\t"}-3.0
        ▁ab${"\t"}-1.0
        a${"\t"}-2.0
        b${"\t"}-2.0
        ab${"\t"}-1.5
        ▁a${"\t"}-1.8
    """.trimIndent()

    private fun tokenizer() = UnigramTokenizer.load(BufferedReader(StringReader(vocabTsv)))

    @Test
    fun viterbi_picksBestScoringSegmentation() {
        // "ab" → 정규화 "▁ab". 후보: [▁ab](-1.0) vs [▁a,b](-3.8) vs [▁,ab](-4.5) → 단일 조각 승리
        assertArrayEquals(intArrayOf(1, 4, 2), tokenizer().encode("ab", 512))
    }

    @Test
    fun multiWord_usesMetaspacePerWord() {
        // "ab ab" → "▁ab▁ab" → [▁ab, ▁ab]
        assertArrayEquals(intArrayOf(1, 4, 4, 2), tokenizer().encode("ab ab", 512))
    }

    @Test
    fun unknownChars_mapToSingleMergedUnk() {
        // "xy" → "▁xy" → [▁, unk(x), unk(y)] → 연속 unk 병합 → [▁, unk]
        assertArrayEquals(intArrayOf(1, 3, 0, 2), tokenizer().encode("xy", 512))
    }

    @Test
    fun truncation_respectsMaxTokens() {
        // "ab ab ab" → body [▁ab,▁ab,▁ab] 인데 maxTokens=4 → body 2개로 절단
        assertArrayEquals(intArrayOf(1, 4, 4, 2), tokenizer().encode("ab ab ab", 4))
    }

    @Test
    fun blankInput_returnsBosEosOnly() {
        assertArrayEquals(intArrayOf(1, 2), tokenizer().encode("   \n ", 512))
    }

    @Test
    fun whitespaceRuns_collapseToSingleMetaspace() {
        assertArrayEquals(
            tokenizer().encode("ab ab", 512),
            tokenizer().encode("ab \n\t ab", 512),
        )
    }

    @Test
    fun invalidHeader_throws() {
        assertThrows(IllegalArgumentException::class.java) {
            UnigramTokenizer.load(BufferedReader(StringReader("garbage\nfoo\t1.0")))
        }
    }
}
