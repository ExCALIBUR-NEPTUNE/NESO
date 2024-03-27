#pragma once
#include <cmath>

// Templated implimentation
// of power function with addition-chain exponentiation
// https://en.wikipedia.org/wiki/Addition-chain_exponentiation
// Usage:
//       a^N = power<T,N>::_(a)
// have to put things in static functions inside structs
// to get around no partial specialisation of functions
// hence the funky syntax

namespace NESO::Basis::Private
{

// base case uses exponentiation by squaring
template <typename T, int N>
struct power {
    static constexpr T
    _(T a)
    {
        static_assert(N >= 0, "N must be non-negative");
        // let the specialisation cover the 0 case
        if constexpr (N & 1) // is odd
            return a * power<T, (N - 1) / 2>::_(a * a);
        else
            return power<T, N / 2>::_(a * a);
    }
};

template <typename T>
struct power<T, 0> {
    static constexpr T
    _([[maybe_unused]] T a)
    {
        return T(1.0);
    }
};

template <typename T>
struct power<T, 1> {
    static constexpr T
    _(T a)
    {
        return a;
    }
};

template <typename T>
struct power<T, 2> {
    static constexpr T
    _(T a)
    {
        return a * a;
    }
};

template <typename T>
struct power<T, 3> {
    static constexpr T
    _(T a)
    {
        return a * a * a;
    }
};

template <typename T>
struct power<T, 4> {
    static constexpr T
    _(T a)
    {
        return power<T, 2>::_(a * a);
    }
};

template <typename T>
struct power<T, 5> {
    static constexpr T
    _(T a)
    {
        return power<T, 4>::_(a) * a;
    }
};

template <typename T>
struct power<T, 6> {
    static constexpr T
    _(T a)
    {
        return power<T, 3>::_(a * a);
    }
};

template <typename T>
struct power<T, 7> {
    static constexpr T
    _(T a)
    {
        return power<T, 6>::_(a) * a;
    }
};

template <typename T>
struct power<T, 8> {
    static constexpr T
    _(T a)
    {
        a *= a;
        a *= a;
        return a * a;
    }
};

template <typename T>
struct power<T, 9> {
    static constexpr T
    _(T a)
    {
        a = power<T, 3>::_(a);
        return power<T, 3>::_(a);
    }
};

template <typename T>
struct power<T, 10> {
    static constexpr T
    _(T a)
    {
        T b = a * a;
        T d = b * b;
        return d * d * b;
    }
};

template <typename T>
struct power<T, 11> {
    static constexpr T
    _(T a)
    {
        return power<T, 10>::_(a) * a;
    }
};

template <typename T>
struct power<T, 12> {
    static constexpr T
    _(T a)
    {
        T d = power<T, 4>::_(a);
        return power<T, 3>::_(d);
    }
};

template <typename T>
struct power<T, 13> {
    static constexpr T
    _(T a)
    {
        return power<T, 12>::_(a) * a;
    }
};

template <typename T>
struct power<T, 14> {
    static constexpr T
    _(T a)
    {
        T b = a * a;
        T d = b * b;
        return power<T, 3>::_(d) * b;
    }
};

template <typename T>
struct power<T, 15> {
    static constexpr T
    _(T a)
    {
        T e = power<T, 5>::_(a);
        return power<T, 3>::_(e);
    }
};

template <typename T>
struct power<T, 16> {
    static constexpr T
    _(T a)
    {
        T h = power<T, 8>::_(a);
        return h * h;
    }
};

} // namespace Meta
