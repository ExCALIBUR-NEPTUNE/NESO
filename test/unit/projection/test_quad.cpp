#include <gtest/gtest.h>
#include <StdRegions/StdQuadExp.h>
#include <StdRegions/StdExpansion.h>
#include <nektar_interface/projection/algorithm_types.hpp>
#include <nektar_interface/projection/device_data.hpp>

#include <nektar_interface/projection/quad.hpp>
#include <CL/sycl.hpp>
using namespace NESO::Project;
using namespace Nektar;
using namespace Nektar::LibUtilities;
using namespace Nektar::StdRegions;


//TODO: need to be sycl malllc/free
template<typename T, int Ndof>
DeviceData<T> 
create_data(cl::sycl::queue &Q, T val, T x, T y, T z = T{0.0}) {
    
    T *dofs = cl::sycl::malloc_device<T>(Ndof,Q);
    Q.fill(dofs,T{0.0},Ndof).wait();

    int *dof_offsets = cl::sycl::malloc_device<int>(1,Q);
    Q.fill(dof_offsets,0,1).wait();
    
    int *cell_ids = cl::sycl::malloc_device<int>(1,Q);
    Q.fill(cell_ids,0,1).wait();
    
    int *par_per_cell = cl::sycl::malloc_device<int>(1,Q);
    Q.fill(par_per_cell,1,1).wait();
    
    auto positions = cl::sycl::malloc_device<T**>(1,Q);
    auto temp0 = cl::sycl::malloc_device<T*>(3,Q); 
    Q.fill(positions,temp0,1).wait(); 
    T P[3] = {x,y,z};
    T *pointP[3] = {
        cl::sycl::malloc_device<T>(1,Q),
        cl::sycl::malloc_device<T>(1,Q),
        cl::sycl::malloc_device<T>(1,Q)
    };
    
    Q.parallel_for<>(3, [=](cl::sycl::id<1> id) {
        positions[0][id] = pointP[id];
        positions[0][id][0] = P[id];
    }).wait();
#if  0
    auto positions = new T**[1];
    positions[0] = new T*[3];
    for (int i = 0; i < 3; ++i) {
        positions[0][i] = new T[1];
        positions[0][i][0] = P[i];
    }

    auto input = new T**[1];
    input[0] = new T*[1];
    input[0][0] = new T[1];
    input[0][0][0] = val;

#endif
    auto input = cl::sycl::malloc_device<T**>(1,Q);
    auto temp1 = cl::sycl::malloc_device<T*>(1,Q); 
    auto temp2 = cl::sycl::malloc_device<T>(1,Q); 
    Q.fill(input,temp1,1).wait(); 
    Q.fill(temp1,temp2,1).wait();
    Q.fill(temp2,val,1).wait();
    
    return DeviceData<T>(dofs,
            dof_offsets, 1, 1, 
            cell_ids, 
            par_per_cell,
            positions,
            input);
}

template <typename T> void free_data(cl::sycl::queue &Q, DeviceData<T> &data) {
  cl::sycl::free(data.dofs, Q);
  cl::sycl::free(data.dof_offsets, Q);
  cl::sycl::free(data.cell_ids, Q);
  cl::sycl::free(data.par_per_cell, Q);
  for (int i = 0; i < 3; ++i) {
      cl::sycl::free(data.positions[0][i],Q);
  }
  cl::sycl::free(data.positions[0], Q);
  cl::sycl::free(data.positions, Q);

  cl::sycl::free(data.input[0][0], Q);
  cl::sycl::free(data.input[0], Q);
  cl::sycl::free(data.input, Q);
}

TEST(Projection, Quad) {
    cl::sycl::queue Q{cl::sycl::default_selector_v};
    auto data = create_data<double,16>(Q,1.0,0.25,0.25);
    PointsKey pk{16, eGaussLobattoLegendre};
    BasisKey bk{eModified_A,4,pk};
    StdExpansion *Quad = new StdQuadExp{bk,bk};
    
    ThreadPerCell2D::template project<4,double,1,1,eQuad<ThreadPerCell2D>>(data,0,Q).wait();
#if 1
    Array<OneD,double> iarray(16,data.dofs); 
    auto x = Quad->Integral(iarray);
#endif
    EXPECT_EQ(x,1.0);
    free_data<double>(Q,data);
}


