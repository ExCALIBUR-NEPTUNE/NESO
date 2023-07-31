#ifndef __SOLVER_CALLBACK_HANDLER_H_
#define __SOLVER_CALLBACK_HANDLER_H_
#include <functional>
#include <vector>

namespace NESO {

template <typename SOLVER> struct SolverCallback {
  virtual void call(SOLVER *state) = 0;
};

/**
 * TODO
 */
template <typename SOLVER> class SolverCallbackHandler {
protected:
  std::vector<std::function<void(SOLVER *)>> pre_integrate_funcs;
  std::vector<std::function<void(SOLVER *)>> post_integrate_funcs;

  template <typename T>
  inline std::function<void(SOLVER *)>
  get_as_consistent_type(std::function<void(T *)> &func) {
    std::function<void(SOLVER *)> f = std::bind(func, std::placeholders::_1);
    return f;
  }

  template <typename T, typename U>
  inline std::function<void(SOLVER *)> get_as_consistent_type(T func, U &inst) {
    std::function<void(SOLVER *)> f =
        std::bind(func, std::ref(inst), std::placeholders::_1);
    return f;
  }

public:
  /**
   * TODO
   */
  template <typename T>
  inline void register_pre_integrate(std::function<void(T *)> &func) {
    this->pre_integrate_funcs.push_back(this->get_as_consistent_type(func));
  }

  /**
   * TODO
   */
  template <typename T, typename U>
  inline void register_pre_integrate(T func, U &inst) {
    this->pre_integrate_funcs.push_back(
        this->get_as_consistent_type(func, inst));
  }

  /**
   * TODO
   */
  template <typename T>
  inline void register_pre_integrate(SolverCallback<T> &obj) {
    this->pre_integrate_funcs.push_back(
        this->get_as_consistent_type(&SolverCallback<T>::call, obj));
  }

  /**
   * TODO
   */
  template <typename T>
  inline void register_post_integrate(std::function<void(T *)> &func) {
    this->post_integrate_funcs.push_back(this->get_as_consistent_type(func));
  }

  /**
   * TODO
   */
  template <typename T, typename U>
  inline void register_post_integrate(T func, U &inst) {
    this->post_integrate_funcs.push_back(
        this->get_as_consistent_type(func, inst));
  }

  /**
   * TODO
   */
  template <typename T>
  inline void register_post_integrate(SolverCallback<T> &obj) {
    this->post_integrate_funcs.push_back(
        this->get_as_consistent_type(&SolverCallback<T>::call, obj));
  }

  /**
   * TODO
   */
  template <typename T> inline void call_pre_integrate(T *state) {
    for (auto &func : this->pre_integrate_funcs) {
      func(state);
    }
  }

  /**
   * TODO
   */
  template <typename T> inline void call_post_integrate(T *state) {
    for (auto &func : this->post_integrate_funcs) {
      func(state);
    }
  }
};

} // namespace NESO
#endif
