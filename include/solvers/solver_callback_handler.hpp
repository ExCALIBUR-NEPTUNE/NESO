#ifndef __SOLVER_CALLBACK_HANDLER_H_
#define __SOLVER_CALLBACK_HANDLER_H_
#include <functional>
#include <vector>

namespace NESO {

/**
 *  Base class which can be inherited from to create a callback for a solver
 *  class called NameOfSolver.
 *
 *      class Foo: public SolverCallback<NameOfSolver> {
 *         void call(NameOfSolver * state){
 *            // Do something with state
 *         }
 *      }
 *
 *  Deriving from this class is not compulsory to create a callback.
 */
template <typename SOLVER> struct SolverCallback {

  /**
   * Call the callback function with the current state passed as a pointer. The
   * callback may modify the solver (at the callers peril). Note the order in
   * which callbacks are called is undefined.
   *
   * @param[in, out] state Pointer to solver instance.
   */
  virtual void call(SOLVER *state) = 0;
};

/**
 * Class to handle calling callbacks from within a solver. This class can be a
 * member variable of a solver or inherited from by the solver. The class is
 * templated on the solver type which defines the pointer type passed to the
 * callback functions.
 */
template <typename SOLVER> class SolverCallbackHandler {
protected:
  /// Functions to be typically called before an integration step.
  std::vector<std::function<void(SOLVER *)>> pre_integrate_funcs;
  /// Functions to be typically called after an integration step.
  std::vector<std::function<void(SOLVER *)>> post_integrate_funcs;

  /**
   *  Helper function to convert an input function handle to a object which can
   *  be stored on the vector of function handles.
   *
   *  @param[in] func Function handle to process.
   *  @returns standardised function handle.
   */
  inline std::function<void(SOLVER *)>
  get_as_consistent_type(std::function<void(SOLVER *)> func) {
    std::function<void(SOLVER *)> f = std::bind(func, std::placeholders::_1);
    return f;
  }

  /**
   *  Helper function to convert an input function handle to a object which can
   *  be stored on the vector of function handles.
   *
   *  @param[in] func Class::method_name to call as function handle.
   *  @param[in] inst object on which to call method.
   *  @returns standardised function handle.
   */
  template <typename T, typename U>
  inline std::function<void(SOLVER *)> get_as_consistent_type(T func, U &inst) {
    std::function<void(SOLVER *)> f =
        std::bind(func, std::ref(inst), std::placeholders::_1);
    return f;
  }

public:
  /**
   * Register a function to be called before each time integration step. e.g.
   *
   *     SolverCallbackHandler<NameOfSolver> solver_callback_handler;
   *     solver_callback_handler.register_pre_integrate(
   *        std::function<void(NameOfSolver *)>{
   *          [&](NameOfSolver *state) {
   *            // use state
   *           }
   *         }
   *       );
   *     }
   *
   * @param[in] func Function handle to add to callbacks.
   */
  inline void register_pre_integrate(std::function<void(SOLVER *)> func) {
    this->pre_integrate_funcs.push_back(this->get_as_consistent_type(func));
  }

  /**
   * Register a class method to be called before each time integration step.
   * e.g.
   *
   *     struct TestInterface {
   *       void call(NameOfSolver *state) {
   *         // use state
   *       }
   *     };
   *
   *     TestInterface ti;
   *     SolverCallbackHandler<NameOfSolver> solver_callback_handler;
   *     solver_callback_handler.register_pre_integrate(&TestInterface::call,
   *                                                    ti);
   *
   * @param[in] func Function handle to add to callbacks in the form of
   * &ClassName::method_name
   * @param[in] inst Instance of ClassName object on which to call method_name.
   */
  template <typename T, typename U>
  inline void register_pre_integrate(T func, U &inst) {
    this->pre_integrate_funcs.push_back(
        this->get_as_consistent_type(func, inst));
  }

  /**
   * Register a type derived of SolverCallback as a callback. e.g.
   *
   *     struct TestInterface : public SolverCallback<NameOfSolver> {
   *       void call(NameOfSolver *state) {
   *         // use state
   *       }
   *     };
   *
   *     TestInterface ti;
   *     SolverCallbackHandler<NameOfSolver> solver_callback_handler;
   *     solver_callback_handler.register_pre_integrate(ti);
   *
   * @param[in] obj Derived type to add as callback object.
   */
  inline void register_pre_integrate(SolverCallback<SOLVER> &obj) {
    this->pre_integrate_funcs.push_back(
        this->get_as_consistent_type(&SolverCallback<SOLVER>::call, obj));
  }

  /**
   * Register a function to be called after each time integration step. e.g.
   *
   *     SolverCallbackHandler<NameOfSolver> solver_callback_handler;
   *     solver_callback_handler.register_post_integrate(
   *        std::function<void(NameOfSolver *)>{
   *          [&](NameOfSolver *state) {
   *            // use state
   *           }
   *         }
   *       );
   *     }
   *
   * @param[in] func Function handle to add to callbacks.
   */
  inline void register_post_integrate(std::function<void(SOLVER *)> func) {
    this->post_integrate_funcs.push_back(this->get_as_consistent_type(func));
  }

  /**
   * Register a class method to be called after each time integration step. e.g.
   *
   *     struct TestInterface {
   *       void call(NameOfSolver *state) {
   *         // use state
   *       }
   *     };
   *
   *     TestInterface ti;
   *     SolverCallbackHandler<NameOfSolver> solver_callback_handler;
   *     solver_callback_handler.register_post_integrate(&TestInterface::call,
   * ti);
   *
   * @param[in] func Function handle to add to callbacks in the form of
   * &ClassName::method_name
   * @param[in] inst Instance of ClassName object on which to call method_name.
   */
  template <typename T, typename U>
  inline void register_post_integrate(T func, U &inst) {
    this->post_integrate_funcs.push_back(
        this->get_as_consistent_type(func, inst));
  }

  /**
   * Register a type derived of SolverCallback as a callback. e.g.
   *
   *     struct TestInterface : public SolverCallback<NameOfSolver> {
   *       void call(NameOfSolver *state) {
   *         // use state
   *       }
   *     };
   *
   *     TestInterface ti;
   *     SolverCallbackHandler<NameOfSolver> solver_callback_handler;
   *     solver_callback_handler.register_post_integrate(ti);
   *
   * @param[in] obj Derived type to add as callback object.
   */
  inline void register_post_integrate(SolverCallback<SOLVER> &obj) {
    this->post_integrate_funcs.push_back(
        this->get_as_consistent_type(&SolverCallback<SOLVER>::call, obj));
  }

  /**
   * Call all function handles which were registered as pre-integration
   * callbacks.
   *
   * @param[in, out] state Solver state used to call each function handle.
   */
  inline void call_pre_integrate(SOLVER *state) {
    for (auto &func : this->pre_integrate_funcs) {
      func(state);
    }
  }

  /**
   * Call all function handles which were registered as post-integration
   * callbacks.
   *
   * @param[in, out] state Solver state used to call each function handle.
   */
  inline void call_post_integrate(SOLVER *state) {
    for (auto &func : this->post_integrate_funcs) {
      func(state);
    }
  }
};

} // namespace NESO
#endif
