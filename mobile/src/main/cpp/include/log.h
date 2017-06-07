//
// Created by Qingqing Cao on 6/7/17.
//

#ifndef MOBIRNN_LOG_H
#define MOBIRNN_LOG_H
#include <android/log.h>
#ifndef LOG_FLAG
#define LOG_FLAG true
#endif // #ifndef LOG_FLAG

inline void empty(...) {
}

#define LOG(...) \
  (LOG_FLAG ? ((void)__android_log_print(ANDROID_LOG_INFO, "native::", __VA_ARGS__)): ((void) empty(__VA_ARGS__)))

#endif //MOBIRNN_LOG_H
