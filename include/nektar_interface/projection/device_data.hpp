#ifndef _NESO_NEKTAR_INTERFACE_PROJECTION_DEVICE_DATA_HPP
#define _NESO_NEKTAR_INTERFACE_PROJECTION_DEVICE_DATA_HPP
#include "restrict.hpp"
namespace NESO::Project {

struct NoFilter {};
struct ApplyFilter {};

template <typename T, typename Filter> struct DeviceData {};

template <typename T> struct DeviceData<T, NoFilter> {
  ;
  using Filter = NoFilter;
  T *dofs;
  int *dof_offsets;
  int *cell_ids;
  int *par_per_cell;
  T ***positions;
  T ***input;
  int ncells;
  int nrow_max;
  DeviceData(T *dofs_, int *dof_offsets_, int ncells_, int nrow_max_,
             int *cell_ids_, int *par_per_cell_, T ***positions_, T ***input_)
      : dofs{dofs_}, dof_offsets{dof_offsets_}, ncells{ncells_},
        nrow_max{nrow_max_}, cell_ids{cell_ids_}, par_per_cell{par_per_cell_},
        positions{positions_}, input{input_} {}
};

template <typename T> struct DeviceData<T, ApplyFilter> {
  using Filter = ApplyFilter;
  T *dofs;
  int *dof_offsets;
  int *cell_ids;
  int *par_per_cell;
  T ***positions;
  T ***input;
  long const *NESO_RESTRICT const *NESO_RESTRICT const *NESO_RESTRICT filter;
  int ncells;
  int nrow_max;
  DeviceData(T *dofs_, int *dof_offsets_, int ncells_, int nrow_max_,
             int *cell_ids_, int *par_per_cell_, T ***positions_, T ***input_,
             long const * NESO_RESTRICT const *NESO_RESTRICT const *NESO_RESTRICT filter_)
      : dofs{dofs_}, dof_offsets{dof_offsets_}, cell_ids{cell_ids_},
        par_per_cell{par_per_cell_}, positions{positions_}, input{input_},
        filter{filter_}, ncells{ncells_}, nrow_max{nrow_max_} {}
};
} // namespace NESO::Project
#endif
