#pragma once

#define AUTO_SWITCH_one_case(n, return_val, func, args, ...)                   \
  case (n): {                                                                  \
    return_val = func<n, __VA_ARGS__> args;                                    \
    break;                                                                     \
  }

#ifndef AUTO_SWITCH_max_nmode
#define AUTO_SWITCH_max_nmode 8
#endif
#include <cassert>
#include <cstdio>

#define AUTO_SWITCH_one_case_1(return_val, func, args, ...)                    \
  AUTO_SWITCH_one_case(1, return_val, func, args, __VA_ARGS__)
#define AUTO_SWITCH_one_case_2(return_val, func, args, ...)                    \
  AUTO_SWITCH_one_case_1(return_val, func, args, __VA_ARGS__)                  \
      AUTO_SWITCH_one_case(2, return_val, func, args, __VA_ARGS__)
#define AUTO_SWITCH_one_case_3(return_val, func, args, ...)                    \
  AUTO_SWITCH_one_case_2(return_val, func, args, __VA_ARGS__)                  \
      AUTO_SWITCH_one_case(3, return_val, func, args, __VA_ARGS__)
#define AUTO_SWITCH_one_case_4(return_val, func, args, ...)                    \
  AUTO_SWITCH_one_case_3(return_val, func, args, __VA_ARGS__)                  \
      AUTO_SWITCH_one_case(4, return_val, func, args, __VA_ARGS__)
#define AUTO_SWITCH_one_case_5(return_val, func, args, ...)                    \
  AUTO_SWITCH_one_case_4(return_val, func, args, __VA_ARGS__)                  \
      AUTO_SWITCH_one_case(5, return_val, func, args, __VA_ARGS__)
#define AUTO_SWITCH_one_case_6(return_val, func, args, ...)                    \
  AUTO_SWITCH_one_case_5(return_val, func, args, __VA_ARGS__)                  \
      AUTO_SWITCH_one_case(6, return_val, func, args, __VA_ARGS__)
#define AUTO_SWITCH_one_case_7(return_val, func, args, ...)                    \
  AUTO_SWITCH_one_case_6(return_val, func, args, __VA_ARGS__)                  \
      AUTO_SWITCH_one_case(7, return_val, func, args, __VA_ARGS__)
#define AUTO_SWITCH_one_case_8(return_val, func, args, ...)                    \
  AUTO_SWITCH_one_case_7(return_val, func, args, __VA_ARGS__)                  \
      AUTO_SWITCH_one_case(8, return_val, func, args, __VA_ARGS__)
#if 0
#define AUTO_SWITCH_one_case_9(return_val, func, args, ...)                    \
  AUTO_SWITCH_one_case_8(return_val, func, args, __VA_ARGS__)                  \
      AUTO_SWITCH_one_case(9, return_val, func, args, __VA_ARGS__)
#define AUTO_SWITCH_one_case_10(return_val, func, args, ...)                   \
  AUTO_SWITCH_one_case_9(return_val, func, args, __VA_ARGS__)                  \
      AUTO_SWITCH_one_case(10, return_val, func, args, __VA_ARGS__)
#define AUTO_SWITCH_one_case_11(return_val, func, args, ...)                   \
  AUTO_SWITCH_one_case_10(return_val, func, args, __VA_ARGS__)                 \
      AUTO_SWITCH_one_case(11, return_val, func, args, __VA_ARGS__)
#define AUTO_SWITCH_one_case_12(return_val, func, args, ...)                   \
  AUTO_SWITCH_one_case_11(return_val, func, args, __VA_ARGS__)                 \
      AUTO_SWITCH_one_case(12, return_val, func, args, __VA_ARGS__)
#define AUTO_SWITCH_one_case_13(return_val, func, args, ...)                   \
  AUTO_SWITCH_one_case_12(return_val, func, args, __VA_ARGS__)                 \
      AUTO_SWITCH_one_case(13, return_val, func, args, __VA_ARGS__)
#define AUTO_SWITCH_one_case_14(return_val, func, args, ...)                   \
  AUTO_SWITCH_one_case_13(return_val, func, args, __VA_ARGS__)                 \
      AUTO_SWITCH_one_case(14, return_val, func, args, __VA_ARGS__)
#define AUTO_SWITCH_one_case_15(return_val, func, args, ...)                   \
  AUTO_SWITCH_one_case_14(return_val, func, args, __VA_ARGS__)                 \
      AUTO_SWITCH_one_case(15, return_val, func, args, __VA_ARGS__)
#define AUTO_SWITCH_one_case_16(return_val, func, args, ...)                   \
  AUTO_SWITCH_one_case_15(return_val, func, args, __VA_ARGS__)                 \
      AUTO_SWITCH_one_case(16, return_val, func, args, __VA_ARGS__)
#define AUTO_SWITCH_one_case_17(return_val, func, args, ...)                   \
  AUTO_SWITCH_one_case_16(return_val, func, args, __VA_ARGS__)                 \
      AUTO_SWITCH_one_case(17, return_val, func, args, __VA_ARGS__)
#define AUTO_SWITCH_one_case_18(return_val, func, args, ...)                   \
  AUTO_SWITCH_one_case_17(return_val, func, args, __VA_ARGS__)                 \
      AUTO_SWITCH_one_case(18, return_val, func, args, __VA_ARGS__)
#define AUTO_SWITCH_one_case_19(return_val, func, args, ...)                   \
  AUTO_SWITCH_one_case_18(return_val, func, args, __VA_ARGS__)                 \
      AUTO_SWITCH_one_case(19, return_val, func, args, __VA_ARGS__)
#endif
#define AUTO_SWITCH_internal(n, var, return_val, func, args, ...)              \
  switch (var) {                                                               \
    AUTO_SWITCH_one_case_##n(return_val, func, args, __VA_ARGS__);             \
  default: {                                                                   \
    fprintf(stderr, "Not supported nmode == %d\n", var);                       \
    assert(false);                                                             \
  }                                                                            \
  };

#define AUTO_SWITCH_internal_(n, var, return_val, func, args, ...)             \
  AUTO_SWITCH_internal(n, var, return_val, func, args, __VA_ARGS__)

#define FUNCTION_ARGS(...) (__VA_ARGS__)
#define AUTO_SWITCH(var, return_val, func, args, ...)                          \
  AUTO_SWITCH_internal_(AUTO_SWITCH_max_nmode, var, return_val, func, args,    \
                        __VA_ARGS__)

/*
 * USAGE:
 *
 * requires some templated function
 * where we want to do a switch on the first argument + an arbitery number of
 * others (actually has to be at least one other or it will break, but if its
 * the only argument then the whole thing can be easier)
 *
 * <int N,M,P,Q,...>
 * fun(a,b,c,f,...) {...}
 *
 * then cat create a switch case statemnt up
 * to AUTO_SWITCH_max_nmode on a variable
 * n like this:
 *
 * AUTO_SWITCH(n,return_val,func,FUNCTION_ARGS(a,b,c,f,...),M,N,P,Q);
 *
 * this will expande to something like
 * switch(n) {
 * case 1:
 *   return_val = func<1,M,N,P,Q>(a,b,c,d,e,f,..);
 *   break;
 * case 2:
 *   return_val = func<1,M,N,P,Q>(a,b,c,d,e,f,..);
 *   break;
 * etc etc
 * case 19:
 *   return_val = func<19,M,N,P,Q>(a,b,c,d,e,f,..);
 *   break;*
 * default:
 *   assert("n is too big")*
 * }
 *
 * NOTE: We have to shove the function arguments in the FUNCTION_ARGS macro
 * or we can just use (). it makes no difference i.e.
 *
 * AUTO_SWITCH(n,func,(a,b,c,f,...),M,N,P,Q);
 *
 * is equivilent to the above, but we need the parenthases or the everything
 * gets swollowed up by the __VA_ARGS__ at the end
 *
 */
