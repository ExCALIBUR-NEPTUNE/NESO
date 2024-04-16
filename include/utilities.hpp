
/**
 * Return the sign (+1, 1, 0) of a value as an int.
 * The possible return values are (+1, -1, 0).
 */
template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}
