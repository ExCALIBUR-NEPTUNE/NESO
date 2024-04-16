#pragma once

namespace NESO::Project {
// Shape does nothing but going to use as tag for function overloading
// should clean things up a bit
template <typename T, typename SHAPE> struct DeviceData {
  T *dofs;
  int *dof_offsets;
  int ncells;
  int nrow_max;
  int *cell_ids;
  int *par_per_cell;
  T ***positions;
  T ***input;
  DeviceData(T *dofs_, int *dof_offsets_, int ncells_, int nrow_max_,
             int *cell_ids_, int *par_per_cell_, T ***positions_, T ***input_)
      : dofs{dofs_}, dof_offsets{dof_offsets_}, ncells{ncells_},
        nrow_max{nrow_max_}, cell_ids{cell_ids_},
        par_per_cell{par_per_cell_}, positions{positions_}, input{input_} {}
};
} // namespace NESO::Project
