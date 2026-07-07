#include <jni.h>

#include <functional>
#include <string>

#include "llm_engine.h"

// com.example.rag_library.NativeLlm 의 @JvmStatic external 구현.
// 심볼: Java_com_example_rag_1library_NativeLlm_<메서드> (@JvmStatic → jclass)
//
// 텍스트는 jbyteArray(표준 UTF-8) 로 주고받는다 — jstring 의 Modified UTF-8 과 달리
// 이모지 등 보충 평면 문자도 손실 없이 전달된다. 경로만 jstring 을 쓴다(BMP 문자는 동일).

namespace {

rag::LlmEngine* llmFromHandle(jlong handle) {
    return reinterpret_cast<rag::LlmEngine*>(handle);
}

void llmThrow(JNIEnv* env, const char* className, const char* msg) {
    if (jclass cls = env->FindClass(className)) {
        env->ThrowNew(cls, msg);
    }
}

bool byteArrayToString(JNIEnv* env, jbyteArray arr, std::string& out) {
    const jsize len = env->GetArrayLength(arr);
    out.resize(static_cast<size_t>(len));
    if (len > 0) {
        env->GetByteArrayRegion(arr, 0, len, reinterpret_cast<jbyte*>(&out[0]));
    }
    return !env->ExceptionCheck();
}

jbyteArray stringToByteArray(JNIEnv* env, const std::string& s) {
    jbyteArray arr = env->NewByteArray(static_cast<jsize>(s.size()));
    if (arr == nullptr) {
        return nullptr;  // OOM — 예외 pending
    }
    if (!s.empty()) {
        env->SetByteArrayRegion(arr, 0, static_cast<jsize>(s.size()),
                                reinterpret_cast<const jbyte*>(s.data()));
    }
    return arr;
}

}  // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_com_example_rag_1library_NativeLlm_nativeCreate(
        JNIEnv* env, jclass, jstring path, jint nCtx, jint nThreads,
        jfloat temperature, jint topK, jfloat topP) {
    const char* p = env->GetStringUTFChars(path, nullptr);
    if (p == nullptr) {
        return 0;
    }
    rag::LlmEngine::Params params;
    params.nCtx = nCtx;
    params.nThreads = nThreads;
    params.temperature = temperature;
    params.topK = topK;
    params.topP = topP;
    rag::LlmEngine* engine = rag::LlmEngine::create(p, params, /*vocabOnly=*/false);
    env->ReleaseStringUTFChars(path, p);
    if (engine == nullptr) {
        llmThrow(env, "java/io/IOException", "failed to load GGUF model");
        return 0;
    }
    return reinterpret_cast<jlong>(engine);
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_rag_1library_NativeLlm_nativeDestroy(JNIEnv*, jclass, jlong handle) {
    delete llmFromHandle(handle);
}

extern "C" JNIEXPORT jint JNICALL
Java_com_example_rag_1library_NativeLlm_nativeCountTokens(
        JNIEnv* env, jclass, jlong handle, jbyteArray textUtf8) {
    rag::LlmEngine* llm = llmFromHandle(handle);
    if (llm == nullptr) {
        llmThrow(env, "java/lang/IllegalStateException", "llm handle is closed");
        return 0;
    }
    std::string text;
    if (!byteArrayToString(env, textUtf8, text)) {
        return 0;
    }
    return llm->countTokens(text);
}

extern "C" JNIEXPORT jbyteArray JNICALL
Java_com_example_rag_1library_NativeLlm_nativeGenerate(
        JNIEnv* env, jclass, jlong handle, jbyteArray promptUtf8, jint maxTokens,
        jobject callback) {
    rag::LlmEngine* llm = llmFromHandle(handle);
    if (llm == nullptr) {
        llmThrow(env, "java/lang/IllegalStateException", "llm handle is closed");
        return nullptr;
    }
    std::string prompt;
    if (!byteArrayToString(env, promptUtf8, prompt)) {
        return nullptr;
    }

    jmethodID onPieceMethod = nullptr;
    if (callback != nullptr) {
        jclass cls = env->GetObjectClass(callback);
        onPieceMethod = env->GetMethodID(cls, "onPiece", "([B)Z");
        env->DeleteLocalRef(cls);
        if (onPieceMethod == nullptr) {
            return nullptr;  // NoSuchMethodError pending
        }
    }

    // 생성은 호출 자바 스레드에서 블로킹 실행되므로 env 를 콜백에서 그대로 써도 안전하다.
    std::function<bool(const std::string&)> onPiece;
    if (callback != nullptr) {
        onPiece = [env, callback, onPieceMethod](const std::string& piece) -> bool {
            jbyteArray arr = stringToByteArray(env, piece);
            if (arr == nullptr) {
                return false;  // OOM — 예외 pending, 생성 중단
            }
            const jboolean keep = env->CallBooleanMethod(callback, onPieceMethod, arr);
            env->DeleteLocalRef(arr);
            if (env->ExceptionCheck()) {
                return false;  // 콜백이 던진 예외 — 생성 중단 후 자바로 전파
            }
            return keep == JNI_TRUE;
        };
    }

    const std::string out = llm->generate(prompt, maxTokens, onPiece);
    if (env->ExceptionCheck()) {
        return nullptr;  // 콜백 예외를 그대로 전파 (pending 상태로 JNI 호출 금지)
    }
    return stringToByteArray(env, out);
}
