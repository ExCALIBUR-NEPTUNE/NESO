#ifndef NEPTUNE_FFTFFTW_H
#define NEPTUNE_FFTFFTW_H

#include "custom_types.hpp"
#include <CL/sycl.hpp>
#include <fftw3.h>
#include <vector>

class FFT {

private:
  inline void generic_copy(sycl::buffer<Complex, 1> &out_b,
                           sycl::buffer<Complex, 1> &in_b) {
    Q.submit([&](sycl::handler &cgh) {

       auto in_a = in_b.get_access<sycl::access::mode::read>(cgh);
       auto out_a = out_b.get_access<sycl::access::mode::write>(cgh);
       cgh.parallel_for<class copy_to_buffer_k>(
           sycl::range<1>{size_t(this->N)}, [=](cl::sycl::id<1> id) {
             reinterpret_cast<double(&)[2]>(out_a[id])[0] =
                 reinterpret_cast<const double(&)[2]>(in_a[id])[0];
             reinterpret_cast<double(&)[2]>(out_a[id])[1] =
                 reinterpret_cast<const double(&)[2]>(in_a[id])[1];
           });
     }).wait();
  }

  inline void copy_to_buffer(sycl::buffer<Complex, 1> &out_b,
                             std::vector<Complex> &in_vec) {
    sycl::buffer<Complex, 1> in_b(in_vec.data(), sycl::range<1>{in_vec.size()});
    generic_copy(out_b, in_b);
  }

  inline void copy_from_buffer(std::vector<Complex> &out_vec,
                               sycl::buffer<Complex, 1> &in_b) {
    sycl::buffer<Complex, 1> out_b(out_vec.data(),
                                   sycl::range<1>{out_vec.size()});
    generic_copy(out_b, in_b);
  }

public:
  sycl::queue &Q;
  // Size of FFT
  int N;

  fftw_plan plan_forward;
  fftw_plan plan_inverse;

  // Real space array
  std::vector<Complex> in;
  // Transformed space array
  std::vector<Complex> out;

  FFT(sycl::queue &Q_in, int N_in) : Q(Q_in), N(N_in) {
    // alloc host space for fftw
    in = std::vector<Complex>(N);
    out = std::vector<Complex>(N);

    // make plan
    plan_forward =
        fftw_plan_dft_1d(N, reinterpret_cast<fftw_complex *>(in.data()),
                         reinterpret_cast<fftw_complex *>(out.data()),
                         FFTW_FORWARD, FFTW_ESTIMATE);
    plan_inverse =
        fftw_plan_dft_1d(N, reinterpret_cast<fftw_complex *>(in.data()),
                         reinterpret_cast<fftw_complex *>(out.data()),
                         FFTW_BACKWARD, FFTW_ESTIMATE);
  }
  ~FFT() {
    fftw_destroy_plan(plan_forward);
    fftw_destroy_plan(plan_inverse);
  };

  void forward(sycl::buffer<Complex, 1> &in_d,
               sycl::buffer<Complex, 1> &out_d) {
    copy_from_buffer(in, in_d);
    fftw_execute(plan_forward);
    copy_to_buffer(out_d, out);
  }

  void backward(sycl::buffer<Complex, 1> &in_d,
                sycl::buffer<Complex, 1> &out_d) {
    copy_from_buffer(in, in_d);
    fftw_execute(plan_inverse);
    copy_to_buffer(out_d, out);
  }
};

#endif
