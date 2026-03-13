#define _STR(x)           #x
#define _PTX_META(s)      asm volatile("// @META:3 " s)

#define META_LOOP(label, min_iters, max_iters, unrolled) \
    _PTX_META("LOOP " _STR(label) " " _STR(min_iters) " " _STR(max_iters) " " _STR(unrolled))