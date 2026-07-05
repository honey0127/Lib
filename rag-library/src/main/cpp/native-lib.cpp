#include <jni.h>
#include <string>

// JNI 심볼 규칙: Java_<패키지경로(_구분)>_<클래스>_<메서드>
// 실제 Kotlin 패키지는 com.example.rag_library 이며, 패키지명 안의 '_'는 '_1'로 이스케이프된다.
//   com.example.rag_library.OnDeviceRAGCore#stringFromJNI
//   → Java_com_example_rag_1library_OnDeviceRAGCore_stringFromJNI
// 이 이름이 한 글자라도 어긋나면 런타임에 UnsatisfiedLinkError 가 발생한다.
extern "C" JNIEXPORT jstring JNICALL
Java_com_example_rag_1library_OnDeviceRAGCore_stringFromJNI(
        JNIEnv* env,
        jobject /* this */) {
    std::string hello = "Hello from C++ Vector Index Core";
    return env->NewStringUTF(hello.c_str());
}