#include <gtest/gtest.h>
#include <neso_particles.hpp>
#include <solvers/solver_callback_handler.hpp>

using namespace NESO;
using namespace NESO::Particles;

namespace {

struct TestClass {
  int ia;
  double da;
};

struct TestFunc {
  double da;
  void operator()(TestClass *test_class) { this->da = test_class->da; }
  void call(TestClass *test_class) { this->da = test_class->da; }
};

struct TestInterface : public SolverCallback<TestClass> {
  double da;
  double ia;
  void call(TestClass *state) {
    this->da = state->da;
    this->ia = state->ia;
  }
};

} // namespace

TEST(SolverCallback, Base) {

  SolverCallbackHandler<TestClass> sc;

  TestFunc test_func_0;
  sc.register_pre_integrate(&TestFunc::call, test_func_0);

  int tia = -1;
  std::function<void(TestClass *)> lambda_func_0 = [&](TestClass *test_class) {
    tia = test_class->ia;
  };

  sc.register_post_integrate(lambda_func_0);

  TestClass tc{};
  tc.ia = 42;
  tc.da = 3.1415;
  sc.call_pre_integrate(&tc);
  ASSERT_EQ(test_func_0.da, 3.1415);

  sc.call_post_integrate(&tc);
  ASSERT_EQ(tia, 42);
}

TEST(SolverCallback, Scope) {

  SolverCallbackHandler<TestClass> sc;

  int tia = -1;

  {
    sc.register_post_integrate(std::function<void(TestClass *)>{
        [&](TestClass *test_class) { tia = test_class->ia; }});
  }

  TestClass tc{};
  tc.ia = 42;
  tc.da = 3.1415;

  sc.call_post_integrate(&tc);
  ASSERT_EQ(tia, 42);
}

TEST(SolverCallback, Class) {

  SolverCallbackHandler<TestClass> sc;
  TestClass tc{};
  tc.ia = 42;
  tc.da = 3.1415;
  TestInterface test_func_0;
  sc.register_pre_integrate(test_func_0);
  sc.call_pre_integrate(&tc);
  ASSERT_EQ(test_func_0.da, 3.1415);

  TestInterface test_func_1;
  sc.register_post_integrate(test_func_1);
  sc.call_post_integrate(&tc);
  ASSERT_EQ(test_func_1.da, 3.1415);
  ASSERT_EQ(test_func_1.ia, 42);
}
