#include "fft_wrappers.hpp"
#include <cmath>
#include <complex>
#include <gtest/gtest.h>
#include <random>

TEST(FFTMKLTest, FFT) {

  auto asyncHandler = [&](sycl::exception_list exceptionList) {};
  auto q = sycl::queue{sycl::default_selector{}, asyncHandler};
  FFT f(q, 16);
  EXPECT_EQ(f.N, 16);
}

void initialize_zero(sycl::queue &q, sycl::buffer<Complex, 1> &x_buf,
                     const int Nk) {

  q.submit([&](sycl::handler &h) {
     sycl::accessor x_acc{x_buf, h, sycl::write_only};
     h.parallel_for<>(sycl::range{size_t(Nk)},
                      [=](sycl::id<1> idx) { x_acc[idx] = 0.0; });
   }).wait();
}

void set_mode_k(sycl::queue &q, sycl::buffer<Complex, 1> &x_buf,
                const int k_ind, const double value) {

  q.submit([&](sycl::handler &h) {
     sycl::accessor x_acc{x_buf, h, sycl::write_only};
     h.single_task<>([=]() {
       x_acc[k_ind].real(value);
       x_acc[k_ind].imag(0.0);
     });
   }).wait();
}

TEST(FFTMKLTest, ForwardSingleModes) {

  auto asyncHandler = [&](sycl::exception_list exceptionList) {};
  auto q = sycl::queue{sycl::default_selector{}, asyncHandler};

  int N = 32;
  FFT f(q, N);

  double *x, *k;
  x = new double[N];
  k = new double[N];
  for (int i = 0; i < N; i++) {
    x[i] = double(i) / double(N);
    k[i] = 2.0 * M_PI * double(i);
  }

  auto in_d = sycl::malloc_device<Complex>(f.N, q);
  sycl::buffer<Complex, 1> in_b(in_d, sycl::range<1>(f.N));

  for (int k_ind = 0; k_ind < f.N; k_ind++) {

    initialize_zero(q, in_b, f.N);
    set_mode_k(q, in_b, k_ind, 1.0);

    // NB: This uses a local definition of in_d. Defining it in the
    // scope below forces an update of the buffer out when out_d
    // goes out of scope.
    std::vector<Complex> out(f.N);
    {
      sycl::buffer<Complex, 1> out_d(out.data(), sycl::range<1>(out.size()));
      f.forward(in_b, out_d);
    }

    for (int i = 0; i < f.N; i++) {
      // std::cout << f.out[2*i] << " " << f.out[2*i+1] << " " <<
      // cos(k[k_ind]*x[i]) << " " << -sin(k[k_ind]*x[i]) <<  "\n";
      ASSERT_NEAR(out.at(i).real(), cos(k[k_ind] * x[i]), 1e-8);
      ASSERT_NEAR(out.at(i).imag(), -sin(k[k_ind] * x[i]), 1e-8);
    }
  }

  sycl::free(in_d, q);
}

TEST(FFTMKLTest, BackwardSingleModes) {

  auto asyncHandler = [&](sycl::exception_list exceptionList) {};
  auto q = sycl::queue{sycl::default_selector{}, asyncHandler};

  int N = 8;
  FFT f(q, N);

  double *x, *k;
  x = new double[N];
  k = new double[N];
  for (int i = 0; i < N; i++) {
    x[i] = double(i) / double(N);
    k[i] = 2.0 * M_PI * double(i);
  }

  auto out_d = sycl::malloc_device<Complex>(f.N, q);
  sycl::buffer<Complex, 1> out_b(out_d, sycl::range<1>(f.N));

  for (int k_ind = 0; k_ind < f.N; k_ind++) {

    // std::cout << k_ind << "\n";
    initialize_zero(q, out_b, f.N);
    set_mode_k(q, out_b, k_ind, 1.0);

    std::vector<Complex> in(f.N);
    {
      sycl::buffer<Complex, 1> in_d(in.data(), sycl::range<1>{in.size()});
      f.backward(out_b, in_d);
    }

    for (int i = 0; i < f.N; i++) {
      // std::cout << f.out[2*i] << " " << f.out[2*i+1] << " " <<
      // cos(k[k_ind]*x[i]) << " " << -sin(k[k_ind]*x[i]) <<  "\n";
      ASSERT_NEAR(in.at(i).real(), cos(k[k_ind] * x[i]), 1e-8);
      ASSERT_NEAR(in.at(i).imag(), sin(k[k_ind] * x[i]), 1e-8);
    }
  }
  sycl::free(out_d, q);
}

/*
 * Forward followed by backward transforms should yield array * N
 */
TEST(FFTMKLTest, ForwardInverse) {

  auto asyncHandler = [&](sycl::exception_list exceptionList) {};
  auto q = sycl::queue{sycl::default_selector{}, asyncHandler};

  int N = 16;
  FFT f(q, N);

  std::vector<Complex> result(N);
  std::vector<Complex> in(N);
  std::vector<Complex> out(N);

  // Random input array
  std::default_random_engine generator;
  for (int i = 0; i < f.N; i++) {
    result.at(i).real(
        std::uniform_real_distribution<double>(-1.0, 1.0)(generator));
    result.at(i).imag(0.0);
  }

  {
    sycl::buffer<Complex, 1> in_d(in.data(), sycl::range<1>{in.size()});
    sycl::buffer<Complex, 1> out_d(out.data(), sycl::range<1>{out.size()});
    sycl::buffer<Complex, 1> result_d(result.data(),
                                      sycl::range<1>{result.size()});

    initialize_zero(q, result_d, f.N);
    initialize_zero(q, out_d, f.N);

    f.forward(in_d, out_d);
    f.backward(out_d, in_d);
  }

  for (int i = 0; i < f.N; i++) {
    ASSERT_NEAR(in.at(i).real() / double(f.N), result.at(i).real(), 1e-8);
    ASSERT_NEAR(in.at(i).imag() / double(f.N), result.at(i).imag(), 1e-8);
  }
}
