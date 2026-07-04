package com.example.rag_library

class OnDeviceRAGCore {
    companion object {
        init {
            System.loadLibrary("rag-core")
        }
    }

    external fun stringFromJNI(): String
}