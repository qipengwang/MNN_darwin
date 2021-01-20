//
//  MNNDefine.h
//  MNN
//
//  Created by MNN on 2018/08/09.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNNDefine_h
#define MNNDefine_h

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
#if defined(_MSC_VER)
#include <Windows.h>
#else
#include <sys/time.h>
#endif

#if defined(__APPLE__)
#include <TargetConditionals.h>
#if TARGET_OS_IPHONE
#define MNN_BUILD_FOR_IOS
#endif
#endif

#ifdef MNN_USE_LOGCAT
#include <android/log.h>
#define MNN_ERROR(format, ...) __android_log_print(ANDROID_LOG_ERROR, "MNNJNI", format, ##__VA_ARGS__)
#define MNN_PRINT(format, ...) __android_log_print(ANDROID_LOG_INFO, "MNNJNI", format, ##__VA_ARGS__)
#else
#define MNN_PRINT(format, ...) printf(format, ##__VA_ARGS__)
#define MNN_ERROR(format, ...) printf(format, ##__VA_ARGS__)
#endif

#ifdef DEBUG
#define MNN_ASSERT(x)                                            \
    {                                                            \
        int res = (x);                                           \
        if (!res) {                                              \
            MNN_ERROR("Error for %s, %d\n", __FILE__, __LINE__); \
            assert(res);                                         \
        }                                                        \
    }
#else
#define MNN_ASSERT(x)
#endif

#define FUNC_PRINT(x) MNN_PRINT(#x "=%d in %s, %d \n", x, __func__, __LINE__);
#define FUNC_PRINT_ALL(x, type) MNN_PRINT(#x "=" #type " %" #type " in %s, %d \n", x, __func__, __LINE__);

#define MNN_CHECK(success, log) \
if(!(success)){ \
MNN_ERROR("Check failed: %s ==> %s\n", #success, #log); \
}

#if defined(_MSC_VER)
#if defined(BUILDING_MNN_DLL)
#define MNN_PUBLIC __declspec(dllexport)
#elif defined(USING_MNN_DLL)
#define MNN_PUBLIC __declspec(dllimport)
#else
#define MNN_PUBLIC
#endif
#else
#define MNN_PUBLIC __attribute__((visibility("default")))
#endif

//memory profiler
#ifdef __linux__
#define MNN_MEMORY_PROFILE(extra) \
    {                          \
        char s[200];          \
        sprintf(s, "/proc/%d/statm", getpid()); \
        FILE* fin = fopen(s, "r");              \
        if(fin) {               \
            fgets(s, 100, fin); \
            FILE* fout = fopen("memory_profile.out", "a"); \
            if(fout) {         \
                fprintf(fout, "%s\t%llu\t%s", extra, MNN_TIME(), s);              \
                strcat(s, "\n");           \
                fclose(fout);           \
            }                  \
            fclose(fin);\
        }\
    }
#else
#define MNN_MEMORY_PROFILE(extra)
#endif // MNN_MEMORY_PROFILE

inline uint64_t MNN_TIME() {
    struct timeval Current;
    gettimeofday(&Current, nullptr);
    uint64_t ret_time = Current.tv_sec * 1000000 + Current.tv_usec;
    return ret_time;
}

#endif /* MNNDefine_h */
