#include <jni.h>
#include <string>

// 주의: 패키지명과 클래스명이 실제 Kotlin 레이어와 완벽히 일치해야 합니다.
// 예시 패키지: com.example.rag, 클래스명: OnDeviceRAGCore
extern "C" JNIEXPORT jstring JNICALL
Java_com_example_rag_OnDeviceRAGCore_stringFromJNI(
        JNIEnv* env,
        jobject /* this */) {
    std::string hello = "Hello from C++ Vector Index Core";
    return env->NewStringUTF(hello.c_str());
}