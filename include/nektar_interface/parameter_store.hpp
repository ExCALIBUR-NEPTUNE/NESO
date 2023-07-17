#ifndef __PARAMETER_STORE_H_
#define __PARAMETER_STORE_H_

#include <map>
#include <memory>
#include <neso_particles.hpp>
#include <string>
#include <type_traits>
#include <variant>

using namespace NESO::Particles;

namespace NESO {

/**
 * Class to store key,value pairs where keys are strings and values are either
 * INT or REAL.
 */
class ParameterStore {
protected:
  std::map<std::string, std::variant<INT, REAL>> store;

public:
  /**
   * Store a REAL or INT value against a string key.
   *
   * @param name Key passed as string.
   * @param value REAL or INT value to store against key.
   */
  template <typename T> inline void set(const std::string name, const T value) {
    static_assert((std::is_same_v<T, REAL> || std::is_same_v<T, INT>),
                  "Bad type passed to ParameterStore::set");
    this->store[name] = value;
  }

  /**
   *  Test if key exists in store.
   *
   *  @param name Key to test for.
   *  @returns True if key exists in store.
   */
  inline bool contains(const std::string name) {
    return (bool)this->store.count(name);
  }

  /**
   *  Retrieve a REAL or INT value stored against a given key. If the key is not
   *  found return the passed value as a default.
   *
   *  @param name Key of value to retrieve from the store.
   *  @param value Default value to return if the key is not present in the
   * store.
   */
  template <typename T>
  inline T get(const std::string name, const T value = 0) {
    static_assert((std::is_same_v<T, REAL> || std::is_same_v<T, INT>),
                  "Bad type passed to ParameterStore::get");
    if (this->contains(name)) {
      return std::get<T>(this->store[name]);
    } else {
      return value;
    }
  }
};

typedef std::shared_ptr<ParameterStore> ParameterStoreSharedPtr;

} // namespace NESO

#endif
