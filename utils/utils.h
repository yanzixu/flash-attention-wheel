#ifndef _FA_PRA_UTILS_
#define _FA_PRA_UTILS_

constexpr bool PRINT = true;
constexpr int THREADID = 0;

#define TAIL                                                               \
    print("\n");                                                           \
    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ \n"); \
    }

#define HEAD(var)                       \
    if (PRINT && thread(THREADID, 0))   \
    {                                   \
        print("tid %d :", threadIdx.x); \
        print(#var##" : ");

#define SHOW(var) \
    HEAD(var)     \
    print(var);   \
    TAIL

#define SHOW_TENSOR(var) \
    HEAD(var)            \
    print_tensor(var);   \
    TAIL

#define SHOW_F(var)           \
    HEAD(var)                 \
    print("%f ", float(var)); \
    TAIL

#endif // !_FA_PRA_UTILS_
