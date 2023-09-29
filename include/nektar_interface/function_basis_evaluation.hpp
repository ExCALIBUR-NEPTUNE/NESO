#ifndef __FUNCTION_BASIS_EVALUATION_H_
#define __FUNCTION_BASIS_EVALUATION_H_
#include "coordinate_mapping.hpp"
#include "particle_interface.hpp"
#include <cstdlib>
#include <map>
#include <memory>
#include <neso_particles.hpp>

#include <MultiRegions/DisContField.h>
#include <LocalRegions/QuadExp.h>
#include <LocalRegions/TriExp.h>
#include <StdRegions/StdExpansion2D.h>

#include "basis_evaluation.hpp"
#include "expansion_looping/basis_evaluate_base.hpp"
#include "expansion_looping/expansion_looping.hpp"
#include "expansion_looping/geom_to_expansion_builder.hpp"
#include "special_functions.hpp"

using namespace NESO::Particles;
using namespace Nektar::LocalRegions;
using namespace Nektar::StdRegions;

#include <CL/sycl.hpp>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>

#define VECTOR_LENGTH 8

namespace NESO {

template <size_t NUM_MODES>
inline sycl::vec<REAL, 8> quadrilateral_evaluate_vector(
  const sycl::vec<REAL, 8> eta0,
  const sycl::vec<REAL, 8> eta1,
  const NekDouble * dofs
){
  return sycl::vec<REAL, 8>(0.0);
}
template <>
inline sycl::vec<REAL, VECTOR_LENGTH> quadrilateral_evaluate_vector<4>(
  const sycl::vec<REAL, VECTOR_LENGTH> eta0,
  const sycl::vec<REAL, VECTOR_LENGTH> eta1,
  const NekDouble * dofs
){
  sycl::vec<REAL, VECTOR_LENGTH> P_0_1_1_eta0(1.0);
  sycl::vec<REAL, VECTOR_LENGTH> P_1_1_1_eta0(2.0*eta0);
  sycl::vec<REAL, VECTOR_LENGTH> x0(eta0 - 1);
  sycl::vec<REAL, VECTOR_LENGTH> x1(eta0 + 1);
  sycl::vec<REAL, VECTOR_LENGTH> x2(0.25*x0*x1);
  sycl::vec<REAL, VECTOR_LENGTH> modA_0_eta0(-0.5*x0);
  sycl::vec<REAL, VECTOR_LENGTH> modA_1_eta0(0.5*x1);
  sycl::vec<REAL, VECTOR_LENGTH> modA_2_eta0(-P_0_1_1_eta0*x2);
  sycl::vec<REAL, VECTOR_LENGTH> modA_3_eta0(-P_1_1_1_eta0*x2);
  sycl::vec<REAL, VECTOR_LENGTH> P_0_1_1_eta1(1.0);
  sycl::vec<REAL, VECTOR_LENGTH> P_1_1_1_eta1(2.0*eta1);
  sycl::vec<REAL, VECTOR_LENGTH> x3(eta1 - 1);
  sycl::vec<REAL, VECTOR_LENGTH> x4(eta1 + 1);
  sycl::vec<REAL, VECTOR_LENGTH> x5(0.25*x3*x4);
  sycl::vec<REAL, VECTOR_LENGTH> modA_0_eta1(-0.5*x3);
  sycl::vec<REAL, VECTOR_LENGTH> modA_1_eta1(0.5*x4);
  sycl::vec<REAL, VECTOR_LENGTH> modA_2_eta1(-P_0_1_1_eta1*x5);
  sycl::vec<REAL, VECTOR_LENGTH> modA_3_eta1(-P_1_1_1_eta1*x5);
  sycl::vec<REAL, VECTOR_LENGTH> dof_0(dofs[0]);
  sycl::vec<REAL, VECTOR_LENGTH> dof_1(dofs[1]);
  sycl::vec<REAL, VECTOR_LENGTH> dof_2(dofs[2]);
  sycl::vec<REAL, VECTOR_LENGTH> dof_3(dofs[3]);
  sycl::vec<REAL, VECTOR_LENGTH> dof_4(dofs[4]);
  sycl::vec<REAL, VECTOR_LENGTH> dof_5(dofs[5]);
  sycl::vec<REAL, VECTOR_LENGTH> dof_6(dofs[6]);
  sycl::vec<REAL, VECTOR_LENGTH> dof_7(dofs[7]);
  sycl::vec<REAL, VECTOR_LENGTH> dof_8(dofs[8]);
  sycl::vec<REAL, VECTOR_LENGTH> dof_9(dofs[9]);
  sycl::vec<REAL, VECTOR_LENGTH> dof_10(dofs[10]);
  sycl::vec<REAL, VECTOR_LENGTH> dof_11(dofs[11]);
  sycl::vec<REAL, VECTOR_LENGTH> dof_12(dofs[12]);
  sycl::vec<REAL, VECTOR_LENGTH> dof_13(dofs[13]);
  sycl::vec<REAL, VECTOR_LENGTH> dof_14(dofs[14]);
  sycl::vec<REAL, VECTOR_LENGTH> dof_15(dofs[15]);
  sycl::vec<REAL, VECTOR_LENGTH> eval_eta0_eta1_0(dof_0*modA_0_eta0*modA_0_eta1);
  sycl::vec<REAL, VECTOR_LENGTH> eval_eta0_eta1_1(dof_1*modA_0_eta1*modA_1_eta0);
  sycl::vec<REAL, VECTOR_LENGTH> eval_eta0_eta1_2(dof_2*modA_0_eta1*modA_2_eta0);
  sycl::vec<REAL, VECTOR_LENGTH> eval_eta0_eta1_3(dof_3*modA_0_eta1*modA_3_eta0);
  sycl::vec<REAL, VECTOR_LENGTH> eval_eta0_eta1_4(dof_4*modA_0_eta0*modA_1_eta1);
  sycl::vec<REAL, VECTOR_LENGTH> eval_eta0_eta1_5(dof_5*modA_1_eta0*modA_1_eta1);
  sycl::vec<REAL, VECTOR_LENGTH> eval_eta0_eta1_6(dof_6*modA_1_eta1*modA_2_eta0);
  sycl::vec<REAL, VECTOR_LENGTH> eval_eta0_eta1_7(dof_7*modA_1_eta1*modA_3_eta0);
  sycl::vec<REAL, VECTOR_LENGTH> eval_eta0_eta1_8(dof_8*modA_0_eta0*modA_2_eta1);
  sycl::vec<REAL, VECTOR_LENGTH> eval_eta0_eta1_9(dof_9*modA_1_eta0*modA_2_eta1);
  sycl::vec<REAL, VECTOR_LENGTH> eval_eta0_eta1_10(dof_10*modA_2_eta0*modA_2_eta1);
  sycl::vec<REAL, VECTOR_LENGTH> eval_eta0_eta1_11(dof_11*modA_2_eta1*modA_3_eta0);
  sycl::vec<REAL, VECTOR_LENGTH> eval_eta0_eta1_12(dof_12*modA_0_eta0*modA_3_eta1);
  sycl::vec<REAL, VECTOR_LENGTH> eval_eta0_eta1_13(dof_13*modA_1_eta0*modA_3_eta1);
  sycl::vec<REAL, VECTOR_LENGTH> eval_eta0_eta1_14(dof_14*modA_2_eta0*modA_3_eta1);
  sycl::vec<REAL, VECTOR_LENGTH> eval_eta0_eta1_15(dof_15*modA_3_eta0*modA_3_eta1);
  sycl::vec<REAL, VECTOR_LENGTH> eval_eta0_eta1(eval_eta0_eta1_0 + eval_eta0_eta1_1 + eval_eta0_eta1_10 + eval_eta0_eta1_11 + eval_eta0_eta1_12 + eval_eta0_eta1_13 + eval_eta0_eta1_14 + eval_eta0_eta1_15 + eval_eta0_eta1_2 + eval_eta0_eta1_3 + eval_eta0_eta1_4 + eval_eta0_eta1_5 + eval_eta0_eta1_6 + eval_eta0_eta1_7 + eval_eta0_eta1_8 + eval_eta0_eta1_9);
  return eval_eta0_eta1;
}

template <>
inline sycl::vec<REAL, VECTOR_LENGTH> quadrilateral_evaluate_vector<8>(
  const sycl::vec<REAL, VECTOR_LENGTH> eta0,
  const sycl::vec<REAL, VECTOR_LENGTH> eta1,
  const NekDouble * dofs
){
  sycl::vec<REAL, VECTOR_LENGTH> P_0_1_1_eta0(1.0);
  sycl::vec<REAL, VECTOR_LENGTH> P_1_1_1_eta0(2.0*eta0);
  sycl::vec<REAL, VECTOR_LENGTH> P_2_1_1_eta0(7.5*eta0 + 0.125*(eta0 - 1.0)*(30.0*eta0 - 30.0) - 4.5);
  sycl::vec<REAL, VECTOR_LENGTH> P_3_1_1_eta0(-0.80000000000000004*P_1_1_1_eta0 + 1.8666666666666667*P_2_1_1_eta0*eta0);
  sycl::vec<REAL, VECTOR_LENGTH> P_4_1_1_eta0(-0.83333333333333326*P_2_1_1_eta0 + 1.875*P_3_1_1_eta0*eta0);
  sycl::vec<REAL, VECTOR_LENGTH> P_5_1_1_eta0(-0.8571428571428571*P_3_1_1_eta0 + 1.8857142857142857*P_4_1_1_eta0*eta0);
  sycl::vec<REAL, VECTOR_LENGTH> x0(eta0 - 1);
  sycl::vec<REAL, VECTOR_LENGTH> x1(eta0 + 1);
  sycl::vec<REAL, VECTOR_LENGTH> x2(0.25*x0*x1);
  sycl::vec<REAL, VECTOR_LENGTH> modA_0_eta0(-0.5*x0);
  sycl::vec<REAL, VECTOR_LENGTH> modA_1_eta0(0.5*x1);
  sycl::vec<REAL, VECTOR_LENGTH> modA_2_eta0(-P_0_1_1_eta0*x2);
  sycl::vec<REAL, VECTOR_LENGTH> modA_3_eta0(-P_1_1_1_eta0*x2);
  sycl::vec<REAL, VECTOR_LENGTH> modA_4_eta0(-P_2_1_1_eta0*x2);
  sycl::vec<REAL, VECTOR_LENGTH> modA_5_eta0(-P_3_1_1_eta0*x2);
  sycl::vec<REAL, VECTOR_LENGTH> modA_6_eta0(-P_4_1_1_eta0*x2);
  sycl::vec<REAL, VECTOR_LENGTH> modA_7_eta0(-P_5_1_1_eta0*x2);
  sycl::vec<REAL, VECTOR_LENGTH> P_0_1_1_eta1(1.0);
  sycl::vec<REAL, VECTOR_LENGTH> P_1_1_1_eta1(2.0*eta1);
  sycl::vec<REAL, VECTOR_LENGTH> P_2_1_1_eta1(7.5*eta1 + 0.125*(eta1 - 1.0)*(30.0*eta1 - 30.0) - 4.5);
  sycl::vec<REAL, VECTOR_LENGTH> P_3_1_1_eta1(-0.80000000000000004*P_1_1_1_eta1 + 1.8666666666666667*P_2_1_1_eta1*eta1);
  sycl::vec<REAL, VECTOR_LENGTH> P_4_1_1_eta1(-0.83333333333333326*P_2_1_1_eta1 + 1.875*P_3_1_1_eta1*eta1);
  sycl::vec<REAL, VECTOR_LENGTH> P_5_1_1_eta1(-0.8571428571428571*P_3_1_1_eta1 + 1.8857142857142857*P_4_1_1_eta1*eta1);
  sycl::vec<REAL, VECTOR_LENGTH> x3(eta1 - 1);
  sycl::vec<REAL, VECTOR_LENGTH> x4(eta1 + 1);
  sycl::vec<REAL, VECTOR_LENGTH> x5(0.25*x3*x4);
  sycl::vec<REAL, VECTOR_LENGTH> modA_0_eta1(-0.5*x3);
  sycl::vec<REAL, VECTOR_LENGTH> modA_1_eta1(0.5*x4);
  sycl::vec<REAL, VECTOR_LENGTH> modA_2_eta1(-P_0_1_1_eta1*x5);
  sycl::vec<REAL, VECTOR_LENGTH> modA_3_eta1(-P_1_1_1_eta1*x5);
  sycl::vec<REAL, VECTOR_LENGTH> modA_4_eta1(-P_2_1_1_eta1*x5);
  sycl::vec<REAL, VECTOR_LENGTH> modA_5_eta1(-P_3_1_1_eta1*x5);
  sycl::vec<REAL, VECTOR_LENGTH> modA_6_eta1(-P_4_1_1_eta1*x5);
  sycl::vec<REAL, VECTOR_LENGTH> modA_7_eta1(-P_5_1_1_eta1*x5);
  sycl::vec<REAL, VECTOR_LENGTH> dof_0(dofs[0]);
  sycl::vec<REAL, VECTOR_LENGTH> dof_1(dofs[1]);
  sycl::vec<REAL, VECTOR_LENGTH> dof_2(dofs[2]);
  sycl::vec<REAL, VECTOR_LENGTH> dof_3(dofs[3]);
  sycl::vec<REAL, VECTOR_LENGTH> dof_4(dofs[4]);
  sycl::vec<REAL, VECTOR_LENGTH> dof_5(dofs[5]);
  sycl::vec<REAL, VECTOR_LENGTH> dof_6(dofs[6]);
  sycl::vec<REAL, VECTOR_LENGTH> dof_7(dofs[7]);
  sycl::vec<REAL, VECTOR_LENGTH> dof_8(dofs[8]);
  sycl::vec<REAL, VECTOR_LENGTH> dof_9(dofs[9]);
  sycl::vec<REAL, VECTOR_LENGTH> dof_10(dofs[10]);
  sycl::vec<REAL, VECTOR_LENGTH> dof_11(dofs[11]);
  sycl::vec<REAL, VECTOR_LENGTH> dof_12(dofs[12]);
  sycl::vec<REAL, VECTOR_LENGTH> dof_13(dofs[13]);
  sycl::vec<REAL, VECTOR_LENGTH> dof_14(dofs[14]);
  sycl::vec<REAL, VECTOR_LENGTH> dof_15(dofs[15]);
  sycl::vec<REAL, VECTOR_LENGTH> dof_16(dofs[16]);
  sycl::vec<REAL, VECTOR_LENGTH> dof_17(dofs[17]);
  sycl::vec<REAL, VECTOR_LENGTH> dof_18(dofs[18]);
  sycl::vec<REAL, VECTOR_LENGTH> dof_19(dofs[19]);
  sycl::vec<REAL, VECTOR_LENGTH> dof_20(dofs[20]);
  sycl::vec<REAL, VECTOR_LENGTH> dof_21(dofs[21]);
  sycl::vec<REAL, VECTOR_LENGTH> dof_22(dofs[22]);
  sycl::vec<REAL, VECTOR_LENGTH> dof_23(dofs[23]);
  sycl::vec<REAL, VECTOR_LENGTH> dof_24(dofs[24]);
  sycl::vec<REAL, VECTOR_LENGTH> dof_25(dofs[25]);
  sycl::vec<REAL, VECTOR_LENGTH> dof_26(dofs[26]);
  sycl::vec<REAL, VECTOR_LENGTH> dof_27(dofs[27]);
  sycl::vec<REAL, VECTOR_LENGTH> dof_28(dofs[28]);
  sycl::vec<REAL, VECTOR_LENGTH> dof_29(dofs[29]);
  sycl::vec<REAL, VECTOR_LENGTH> dof_30(dofs[30]);
  sycl::vec<REAL, VECTOR_LENGTH> dof_31(dofs[31]);
  sycl::vec<REAL, VECTOR_LENGTH> dof_32(dofs[32]);
  sycl::vec<REAL, VECTOR_LENGTH> dof_33(dofs[33]);
  sycl::vec<REAL, VECTOR_LENGTH> dof_34(dofs[34]);
  sycl::vec<REAL, VECTOR_LENGTH> dof_35(dofs[35]);
  sycl::vec<REAL, VECTOR_LENGTH> dof_36(dofs[36]);
  sycl::vec<REAL, VECTOR_LENGTH> dof_37(dofs[37]);
  sycl::vec<REAL, VECTOR_LENGTH> dof_38(dofs[38]);
  sycl::vec<REAL, VECTOR_LENGTH> dof_39(dofs[39]);
  sycl::vec<REAL, VECTOR_LENGTH> dof_40(dofs[40]);
  sycl::vec<REAL, VECTOR_LENGTH> dof_41(dofs[41]);
  sycl::vec<REAL, VECTOR_LENGTH> dof_42(dofs[42]);
  sycl::vec<REAL, VECTOR_LENGTH> dof_43(dofs[43]);
  sycl::vec<REAL, VECTOR_LENGTH> dof_44(dofs[44]);
  sycl::vec<REAL, VECTOR_LENGTH> dof_45(dofs[45]);
  sycl::vec<REAL, VECTOR_LENGTH> dof_46(dofs[46]);
  sycl::vec<REAL, VECTOR_LENGTH> dof_47(dofs[47]);
  sycl::vec<REAL, VECTOR_LENGTH> dof_48(dofs[48]);
  sycl::vec<REAL, VECTOR_LENGTH> dof_49(dofs[49]);
  sycl::vec<REAL, VECTOR_LENGTH> dof_50(dofs[50]);
  sycl::vec<REAL, VECTOR_LENGTH> dof_51(dofs[51]);
  sycl::vec<REAL, VECTOR_LENGTH> dof_52(dofs[52]);
  sycl::vec<REAL, VECTOR_LENGTH> dof_53(dofs[53]);
  sycl::vec<REAL, VECTOR_LENGTH> dof_54(dofs[54]);
  sycl::vec<REAL, VECTOR_LENGTH> dof_55(dofs[55]);
  sycl::vec<REAL, VECTOR_LENGTH> dof_56(dofs[56]);
  sycl::vec<REAL, VECTOR_LENGTH> dof_57(dofs[57]);
  sycl::vec<REAL, VECTOR_LENGTH> dof_58(dofs[58]);
  sycl::vec<REAL, VECTOR_LENGTH> dof_59(dofs[59]);
  sycl::vec<REAL, VECTOR_LENGTH> dof_60(dofs[60]);
  sycl::vec<REAL, VECTOR_LENGTH> dof_61(dofs[61]);
  sycl::vec<REAL, VECTOR_LENGTH> dof_62(dofs[62]);
  sycl::vec<REAL, VECTOR_LENGTH> dof_63(dofs[63]);
  sycl::vec<REAL, VECTOR_LENGTH> eval_eta0_eta1_0(dof_0*modA_0_eta0*modA_0_eta1);
  sycl::vec<REAL, VECTOR_LENGTH> eval_eta0_eta1_1(dof_1*modA_0_eta1*modA_1_eta0);
  sycl::vec<REAL, VECTOR_LENGTH> eval_eta0_eta1_2(dof_2*modA_0_eta1*modA_2_eta0);
  sycl::vec<REAL, VECTOR_LENGTH> eval_eta0_eta1_3(dof_3*modA_0_eta1*modA_3_eta0);
  sycl::vec<REAL, VECTOR_LENGTH> eval_eta0_eta1_4(dof_4*modA_0_eta1*modA_4_eta0);
  sycl::vec<REAL, VECTOR_LENGTH> eval_eta0_eta1_5(dof_5*modA_0_eta1*modA_5_eta0);
  sycl::vec<REAL, VECTOR_LENGTH> eval_eta0_eta1_6(dof_6*modA_0_eta1*modA_6_eta0);
  sycl::vec<REAL, VECTOR_LENGTH> eval_eta0_eta1_7(dof_7*modA_0_eta1*modA_7_eta0);
  sycl::vec<REAL, VECTOR_LENGTH> eval_eta0_eta1_8(dof_8*modA_0_eta0*modA_1_eta1);
  sycl::vec<REAL, VECTOR_LENGTH> eval_eta0_eta1_9(dof_9*modA_1_eta0*modA_1_eta1);
  sycl::vec<REAL, VECTOR_LENGTH> eval_eta0_eta1_10(dof_10*modA_1_eta1*modA_2_eta0);
  sycl::vec<REAL, VECTOR_LENGTH> eval_eta0_eta1_11(dof_11*modA_1_eta1*modA_3_eta0);
  sycl::vec<REAL, VECTOR_LENGTH> eval_eta0_eta1_12(dof_12*modA_1_eta1*modA_4_eta0);
  sycl::vec<REAL, VECTOR_LENGTH> eval_eta0_eta1_13(dof_13*modA_1_eta1*modA_5_eta0);
  sycl::vec<REAL, VECTOR_LENGTH> eval_eta0_eta1_14(dof_14*modA_1_eta1*modA_6_eta0);
  sycl::vec<REAL, VECTOR_LENGTH> eval_eta0_eta1_15(dof_15*modA_1_eta1*modA_7_eta0);
  sycl::vec<REAL, VECTOR_LENGTH> eval_eta0_eta1_16(dof_16*modA_0_eta0*modA_2_eta1);
  sycl::vec<REAL, VECTOR_LENGTH> eval_eta0_eta1_17(dof_17*modA_1_eta0*modA_2_eta1);
  sycl::vec<REAL, VECTOR_LENGTH> eval_eta0_eta1_18(dof_18*modA_2_eta0*modA_2_eta1);
  sycl::vec<REAL, VECTOR_LENGTH> eval_eta0_eta1_19(dof_19*modA_2_eta1*modA_3_eta0);
  sycl::vec<REAL, VECTOR_LENGTH> eval_eta0_eta1_20(dof_20*modA_2_eta1*modA_4_eta0);
  sycl::vec<REAL, VECTOR_LENGTH> eval_eta0_eta1_21(dof_21*modA_2_eta1*modA_5_eta0);
  sycl::vec<REAL, VECTOR_LENGTH> eval_eta0_eta1_22(dof_22*modA_2_eta1*modA_6_eta0);
  sycl::vec<REAL, VECTOR_LENGTH> eval_eta0_eta1_23(dof_23*modA_2_eta1*modA_7_eta0);
  sycl::vec<REAL, VECTOR_LENGTH> eval_eta0_eta1_24(dof_24*modA_0_eta0*modA_3_eta1);
  sycl::vec<REAL, VECTOR_LENGTH> eval_eta0_eta1_25(dof_25*modA_1_eta0*modA_3_eta1);
  sycl::vec<REAL, VECTOR_LENGTH> eval_eta0_eta1_26(dof_26*modA_2_eta0*modA_3_eta1);
  sycl::vec<REAL, VECTOR_LENGTH> eval_eta0_eta1_27(dof_27*modA_3_eta0*modA_3_eta1);
  sycl::vec<REAL, VECTOR_LENGTH> eval_eta0_eta1_28(dof_28*modA_3_eta1*modA_4_eta0);
  sycl::vec<REAL, VECTOR_LENGTH> eval_eta0_eta1_29(dof_29*modA_3_eta1*modA_5_eta0);
  sycl::vec<REAL, VECTOR_LENGTH> eval_eta0_eta1_30(dof_30*modA_3_eta1*modA_6_eta0);
  sycl::vec<REAL, VECTOR_LENGTH> eval_eta0_eta1_31(dof_31*modA_3_eta1*modA_7_eta0);
  sycl::vec<REAL, VECTOR_LENGTH> eval_eta0_eta1_32(dof_32*modA_0_eta0*modA_4_eta1);
  sycl::vec<REAL, VECTOR_LENGTH> eval_eta0_eta1_33(dof_33*modA_1_eta0*modA_4_eta1);
  sycl::vec<REAL, VECTOR_LENGTH> eval_eta0_eta1_34(dof_34*modA_2_eta0*modA_4_eta1);
  sycl::vec<REAL, VECTOR_LENGTH> eval_eta0_eta1_35(dof_35*modA_3_eta0*modA_4_eta1);
  sycl::vec<REAL, VECTOR_LENGTH> eval_eta0_eta1_36(dof_36*modA_4_eta0*modA_4_eta1);
  sycl::vec<REAL, VECTOR_LENGTH> eval_eta0_eta1_37(dof_37*modA_4_eta1*modA_5_eta0);
  sycl::vec<REAL, VECTOR_LENGTH> eval_eta0_eta1_38(dof_38*modA_4_eta1*modA_6_eta0);
  sycl::vec<REAL, VECTOR_LENGTH> eval_eta0_eta1_39(dof_39*modA_4_eta1*modA_7_eta0);
  sycl::vec<REAL, VECTOR_LENGTH> eval_eta0_eta1_40(dof_40*modA_0_eta0*modA_5_eta1);
  sycl::vec<REAL, VECTOR_LENGTH> eval_eta0_eta1_41(dof_41*modA_1_eta0*modA_5_eta1);
  sycl::vec<REAL, VECTOR_LENGTH> eval_eta0_eta1_42(dof_42*modA_2_eta0*modA_5_eta1);
  sycl::vec<REAL, VECTOR_LENGTH> eval_eta0_eta1_43(dof_43*modA_3_eta0*modA_5_eta1);
  sycl::vec<REAL, VECTOR_LENGTH> eval_eta0_eta1_44(dof_44*modA_4_eta0*modA_5_eta1);
  sycl::vec<REAL, VECTOR_LENGTH> eval_eta0_eta1_45(dof_45*modA_5_eta0*modA_5_eta1);
  sycl::vec<REAL, VECTOR_LENGTH> eval_eta0_eta1_46(dof_46*modA_5_eta1*modA_6_eta0);
  sycl::vec<REAL, VECTOR_LENGTH> eval_eta0_eta1_47(dof_47*modA_5_eta1*modA_7_eta0);
  sycl::vec<REAL, VECTOR_LENGTH> eval_eta0_eta1_48(dof_48*modA_0_eta0*modA_6_eta1);
  sycl::vec<REAL, VECTOR_LENGTH> eval_eta0_eta1_49(dof_49*modA_1_eta0*modA_6_eta1);
  sycl::vec<REAL, VECTOR_LENGTH> eval_eta0_eta1_50(dof_50*modA_2_eta0*modA_6_eta1);
  sycl::vec<REAL, VECTOR_LENGTH> eval_eta0_eta1_51(dof_51*modA_3_eta0*modA_6_eta1);
  sycl::vec<REAL, VECTOR_LENGTH> eval_eta0_eta1_52(dof_52*modA_4_eta0*modA_6_eta1);
  sycl::vec<REAL, VECTOR_LENGTH> eval_eta0_eta1_53(dof_53*modA_5_eta0*modA_6_eta1);
  sycl::vec<REAL, VECTOR_LENGTH> eval_eta0_eta1_54(dof_54*modA_6_eta0*modA_6_eta1);
  sycl::vec<REAL, VECTOR_LENGTH> eval_eta0_eta1_55(dof_55*modA_6_eta1*modA_7_eta0);
  sycl::vec<REAL, VECTOR_LENGTH> eval_eta0_eta1_56(dof_56*modA_0_eta0*modA_7_eta1);
  sycl::vec<REAL, VECTOR_LENGTH> eval_eta0_eta1_57(dof_57*modA_1_eta0*modA_7_eta1);
  sycl::vec<REAL, VECTOR_LENGTH> eval_eta0_eta1_58(dof_58*modA_2_eta0*modA_7_eta1);
  sycl::vec<REAL, VECTOR_LENGTH> eval_eta0_eta1_59(dof_59*modA_3_eta0*modA_7_eta1);
  sycl::vec<REAL, VECTOR_LENGTH> eval_eta0_eta1_60(dof_60*modA_4_eta0*modA_7_eta1);
  sycl::vec<REAL, VECTOR_LENGTH> eval_eta0_eta1_61(dof_61*modA_5_eta0*modA_7_eta1);
  sycl::vec<REAL, VECTOR_LENGTH> eval_eta0_eta1_62(dof_62*modA_6_eta0*modA_7_eta1);
  sycl::vec<REAL, VECTOR_LENGTH> eval_eta0_eta1_63(dof_63*modA_7_eta0*modA_7_eta1);
  sycl::vec<REAL, VECTOR_LENGTH> eval_eta0_eta1(eval_eta0_eta1_0 + eval_eta0_eta1_1 + eval_eta0_eta1_10 + eval_eta0_eta1_11 + eval_eta0_eta1_12 + eval_eta0_eta1_13 + eval_eta0_eta1_14 + eval_eta0_eta1_15 + eval_eta0_eta1_16 + eval_eta0_eta1_17 + eval_eta0_eta1_18 + eval_eta0_eta1_19 + eval_eta0_eta1_2 + eval_eta0_eta1_20 + eval_eta0_eta1_21 + eval_eta0_eta1_22 + eval_eta0_eta1_23 + eval_eta0_eta1_24 + eval_eta0_eta1_25 + eval_eta0_eta1_26 + eval_eta0_eta1_27 + eval_eta0_eta1_28 + eval_eta0_eta1_29 + eval_eta0_eta1_3 + eval_eta0_eta1_30 + eval_eta0_eta1_31 + eval_eta0_eta1_32 + eval_eta0_eta1_33 + eval_eta0_eta1_34 + eval_eta0_eta1_35 + eval_eta0_eta1_36 + eval_eta0_eta1_37 + eval_eta0_eta1_38 + eval_eta0_eta1_39 + eval_eta0_eta1_4 + eval_eta0_eta1_40 + eval_eta0_eta1_41 + eval_eta0_eta1_42 + eval_eta0_eta1_43 + eval_eta0_eta1_44 + eval_eta0_eta1_45 + eval_eta0_eta1_46 + eval_eta0_eta1_47 + eval_eta0_eta1_48 + eval_eta0_eta1_49 + eval_eta0_eta1_5 + eval_eta0_eta1_50 + eval_eta0_eta1_51 + eval_eta0_eta1_52 + eval_eta0_eta1_53 + eval_eta0_eta1_54 + eval_eta0_eta1_55 + eval_eta0_eta1_56 + eval_eta0_eta1_57 + eval_eta0_eta1_58 + eval_eta0_eta1_59 + eval_eta0_eta1_6 + eval_eta0_eta1_60 + eval_eta0_eta1_61 + eval_eta0_eta1_62 + eval_eta0_eta1_63 + eval_eta0_eta1_7 + eval_eta0_eta1_8 + eval_eta0_eta1_9);
  return eval_eta0_eta1;
}

template <size_t NUM_MODES>
inline REAL quadrilateral_evaluate_scalar(
  const REAL eta0,
  const REAL eta1,
  const NekDouble * dofs
){
  return 0.0;
}

template <>
inline REAL quadrilateral_evaluate_scalar<4>(
  const REAL eta0,
  const REAL eta1,
  const NekDouble * dofs
){
  REAL P_0_1_1_eta0(1.0);
  REAL P_1_1_1_eta0(2.0*eta0);
  REAL x0(eta0 - 1);
  REAL x1(eta0 + 1);
  REAL x2(0.25*x0*x1);
  REAL modA_0_eta0(-0.5*x0);
  REAL modA_1_eta0(0.5*x1);
  REAL modA_2_eta0(-P_0_1_1_eta0*x2);
  REAL modA_3_eta0(-P_1_1_1_eta0*x2);
  REAL P_0_1_1_eta1(1.0);
  REAL P_1_1_1_eta1(2.0*eta1);
  REAL x3(eta1 - 1);
  REAL x4(eta1 + 1);
  REAL x5(0.25*x3*x4);
  REAL modA_0_eta1(-0.5*x3);
  REAL modA_1_eta1(0.5*x4);
  REAL modA_2_eta1(-P_0_1_1_eta1*x5);
  REAL modA_3_eta1(-P_1_1_1_eta1*x5);
  REAL dof_0(dofs[0]);
  REAL dof_1(dofs[1]);
  REAL dof_2(dofs[2]);
  REAL dof_3(dofs[3]);
  REAL dof_4(dofs[4]);
  REAL dof_5(dofs[5]);
  REAL dof_6(dofs[6]);
  REAL dof_7(dofs[7]);
  REAL dof_8(dofs[8]);
  REAL dof_9(dofs[9]);
  REAL dof_10(dofs[10]);
  REAL dof_11(dofs[11]);
  REAL dof_12(dofs[12]);
  REAL dof_13(dofs[13]);
  REAL dof_14(dofs[14]);
  REAL dof_15(dofs[15]);
  REAL eval_eta0_eta1_0(dof_0*modA_0_eta0*modA_0_eta1);
  REAL eval_eta0_eta1_1(dof_1*modA_0_eta1*modA_1_eta0);
  REAL eval_eta0_eta1_2(dof_2*modA_0_eta1*modA_2_eta0);
  REAL eval_eta0_eta1_3(dof_3*modA_0_eta1*modA_3_eta0);
  REAL eval_eta0_eta1_4(dof_4*modA_0_eta0*modA_1_eta1);
  REAL eval_eta0_eta1_5(dof_5*modA_1_eta0*modA_1_eta1);
  REAL eval_eta0_eta1_6(dof_6*modA_1_eta1*modA_2_eta0);
  REAL eval_eta0_eta1_7(dof_7*modA_1_eta1*modA_3_eta0);
  REAL eval_eta0_eta1_8(dof_8*modA_0_eta0*modA_2_eta1);
  REAL eval_eta0_eta1_9(dof_9*modA_1_eta0*modA_2_eta1);
  REAL eval_eta0_eta1_10(dof_10*modA_2_eta0*modA_2_eta1);
  REAL eval_eta0_eta1_11(dof_11*modA_2_eta1*modA_3_eta0);
  REAL eval_eta0_eta1_12(dof_12*modA_0_eta0*modA_3_eta1);
  REAL eval_eta0_eta1_13(dof_13*modA_1_eta0*modA_3_eta1);
  REAL eval_eta0_eta1_14(dof_14*modA_2_eta0*modA_3_eta1);
  REAL eval_eta0_eta1_15(dof_15*modA_3_eta0*modA_3_eta1);
  REAL eval_eta0_eta1(eval_eta0_eta1_0 + eval_eta0_eta1_1 + eval_eta0_eta1_10 + eval_eta0_eta1_11 + eval_eta0_eta1_12 + eval_eta0_eta1_13 + eval_eta0_eta1_14 + eval_eta0_eta1_15 + eval_eta0_eta1_2 + eval_eta0_eta1_3 + eval_eta0_eta1_4 + eval_eta0_eta1_5 + eval_eta0_eta1_6 + eval_eta0_eta1_7 + eval_eta0_eta1_8 + eval_eta0_eta1_9);
  return eval_eta0_eta1;
}





template <>
inline REAL quadrilateral_evaluate_scalar<8>(
  const REAL eta0,
  const REAL eta1,
  const NekDouble * dofs
){
  // 21
  REAL P_0_1_1_eta0(1.0);
  REAL P_1_1_1_eta0(2.0*eta0);
  REAL P_2_1_1_eta0(7.5*eta0 + 0.125*(eta0 - 1.0)*(30*eta0 - 30.0) - 4.5);
  REAL P_3_1_1_eta0(-0.80000000000000004*P_1_1_1_eta0 + 1.8666666666666667*P_2_1_1_eta0*eta0);
  REAL P_4_1_1_eta0(-0.83333333333333326*P_2_1_1_eta0 + 1.875*P_3_1_1_eta0*eta0);
  REAL P_5_1_1_eta0(-0.8571428571428571*P_3_1_1_eta0 + 1.8857142857142857*P_4_1_1_eta0*eta0);
  // 18
  REAL x0(eta0 - 1);
  REAL x1(eta0 + 1);
  REAL x2(0.25*x0*x1);
  REAL modA_0_eta0(-0.5*x0);
  REAL modA_1_eta0(0.5*x1);
  REAL modA_2_eta0(-P_0_1_1_eta0*x2);
  REAL modA_3_eta0(-P_1_1_1_eta0*x2);
  REAL modA_4_eta0(-P_2_1_1_eta0*x2);
  REAL modA_5_eta0(-P_3_1_1_eta0*x2);
  REAL modA_6_eta0(-P_4_1_1_eta0*x2);
  REAL modA_7_eta0(-P_5_1_1_eta0*x2);
  // 21
  REAL P_0_1_1_eta1(1.0);
  REAL P_1_1_1_eta1(2.0*eta1);
  REAL P_2_1_1_eta1(7.5*eta1 + 0.125*(eta1 - 1.0)*(30*eta1 - 30.0) - 4.5);
  REAL P_3_1_1_eta1(-0.80000000000000004*P_1_1_1_eta1 + 1.8666666666666667*P_2_1_1_eta1*eta1);
  REAL P_4_1_1_eta1(-0.83333333333333326*P_2_1_1_eta1 + 1.875*P_3_1_1_eta1*eta1);
  REAL P_5_1_1_eta1(-0.8571428571428571*P_3_1_1_eta1 + 1.8857142857142857*P_4_1_1_eta1*eta1);
  // 18
  REAL x3(eta1 - 1);
  REAL x4(eta1 + 1);
  REAL x5(0.25*x3*x4);
  REAL modA_0_eta1(-0.5*x3);
  REAL modA_1_eta1(0.5*x4);
  REAL modA_2_eta1(-P_0_1_1_eta1*x5);
  REAL modA_3_eta1(-P_1_1_1_eta1*x5);
  REAL modA_4_eta1(-P_2_1_1_eta1*x5);
  REAL modA_5_eta1(-P_3_1_1_eta1*x5);
  REAL modA_6_eta1(-P_4_1_1_eta1*x5);
  REAL modA_7_eta1(-P_5_1_1_eta1*x5);
  //
  REAL dof_0(dofs[0]);
  REAL dof_1(dofs[1]);
  REAL dof_2(dofs[2]);
  REAL dof_3(dofs[3]);
  REAL dof_4(dofs[4]);
  REAL dof_5(dofs[5]);
  REAL dof_6(dofs[6]);
  REAL dof_7(dofs[7]);
  REAL dof_8(dofs[8]);
  REAL dof_9(dofs[9]);
  REAL dof_10(dofs[10]);
  REAL dof_11(dofs[11]);
  REAL dof_12(dofs[12]);
  REAL dof_13(dofs[13]);
  REAL dof_14(dofs[14]);
  REAL dof_15(dofs[15]);
  REAL dof_16(dofs[16]);
  REAL dof_17(dofs[17]);
  REAL dof_18(dofs[18]);
  REAL dof_19(dofs[19]);
  REAL dof_20(dofs[20]);
  REAL dof_21(dofs[21]);
  REAL dof_22(dofs[22]);
  REAL dof_23(dofs[23]);
  REAL dof_24(dofs[24]);
  REAL dof_25(dofs[25]);
  REAL dof_26(dofs[26]);
  REAL dof_27(dofs[27]);
  REAL dof_28(dofs[28]);
  REAL dof_29(dofs[29]);
  REAL dof_30(dofs[30]);
  REAL dof_31(dofs[31]);
  REAL dof_32(dofs[32]);
  REAL dof_33(dofs[33]);
  REAL dof_34(dofs[34]);
  REAL dof_35(dofs[35]);
  REAL dof_36(dofs[36]);
  REAL dof_37(dofs[37]);
  REAL dof_38(dofs[38]);
  REAL dof_39(dofs[39]);
  REAL dof_40(dofs[40]);
  REAL dof_41(dofs[41]);
  REAL dof_42(dofs[42]);
  REAL dof_43(dofs[43]);
  REAL dof_44(dofs[44]);
  REAL dof_45(dofs[45]);
  REAL dof_46(dofs[46]);
  REAL dof_47(dofs[47]);
  REAL dof_48(dofs[48]);
  REAL dof_49(dofs[49]);
  REAL dof_50(dofs[50]);
  REAL dof_51(dofs[51]);
  REAL dof_52(dofs[52]);
  REAL dof_53(dofs[53]);
  REAL dof_54(dofs[54]);
  REAL dof_55(dofs[55]);
  REAL dof_56(dofs[56]);
  REAL dof_57(dofs[57]);
  REAL dof_58(dofs[58]);
  REAL dof_59(dofs[59]);
  REAL dof_60(dofs[60]);
  REAL dof_61(dofs[61]);
  REAL dof_62(dofs[62]);
  REAL dof_63(dofs[63]);
  // 128
  REAL eval_eta0_eta1_0(dof_0*modA_0_eta0*modA_0_eta1);
  REAL eval_eta0_eta1_1(dof_1*modA_0_eta1*modA_1_eta0);
  REAL eval_eta0_eta1_2(dof_2*modA_0_eta1*modA_2_eta0);
  REAL eval_eta0_eta1_3(dof_3*modA_0_eta1*modA_3_eta0);
  REAL eval_eta0_eta1_4(dof_4*modA_0_eta1*modA_4_eta0);
  REAL eval_eta0_eta1_5(dof_5*modA_0_eta1*modA_5_eta0);
  REAL eval_eta0_eta1_6(dof_6*modA_0_eta1*modA_6_eta0);
  REAL eval_eta0_eta1_7(dof_7*modA_0_eta1*modA_7_eta0);
  REAL eval_eta0_eta1_8(dof_8*modA_0_eta0*modA_1_eta1);
  REAL eval_eta0_eta1_9(dof_9*modA_1_eta0*modA_1_eta1);
  REAL eval_eta0_eta1_10(dof_10*modA_1_eta1*modA_2_eta0);
  REAL eval_eta0_eta1_11(dof_11*modA_1_eta1*modA_3_eta0);
  REAL eval_eta0_eta1_12(dof_12*modA_1_eta1*modA_4_eta0);
  REAL eval_eta0_eta1_13(dof_13*modA_1_eta1*modA_5_eta0);
  REAL eval_eta0_eta1_14(dof_14*modA_1_eta1*modA_6_eta0);
  REAL eval_eta0_eta1_15(dof_15*modA_1_eta1*modA_7_eta0);
  REAL eval_eta0_eta1_16(dof_16*modA_0_eta0*modA_2_eta1);
  REAL eval_eta0_eta1_17(dof_17*modA_1_eta0*modA_2_eta1);
  REAL eval_eta0_eta1_18(dof_18*modA_2_eta0*modA_2_eta1);
  REAL eval_eta0_eta1_19(dof_19*modA_2_eta1*modA_3_eta0);
  REAL eval_eta0_eta1_20(dof_20*modA_2_eta1*modA_4_eta0);
  REAL eval_eta0_eta1_21(dof_21*modA_2_eta1*modA_5_eta0);
  REAL eval_eta0_eta1_22(dof_22*modA_2_eta1*modA_6_eta0);
  REAL eval_eta0_eta1_23(dof_23*modA_2_eta1*modA_7_eta0);
  REAL eval_eta0_eta1_24(dof_24*modA_0_eta0*modA_3_eta1);
  REAL eval_eta0_eta1_25(dof_25*modA_1_eta0*modA_3_eta1);
  REAL eval_eta0_eta1_26(dof_26*modA_2_eta0*modA_3_eta1);
  REAL eval_eta0_eta1_27(dof_27*modA_3_eta0*modA_3_eta1);
  REAL eval_eta0_eta1_28(dof_28*modA_3_eta1*modA_4_eta0);
  REAL eval_eta0_eta1_29(dof_29*modA_3_eta1*modA_5_eta0);
  REAL eval_eta0_eta1_30(dof_30*modA_3_eta1*modA_6_eta0);
  REAL eval_eta0_eta1_31(dof_31*modA_3_eta1*modA_7_eta0);
  REAL eval_eta0_eta1_32(dof_32*modA_0_eta0*modA_4_eta1);
  REAL eval_eta0_eta1_33(dof_33*modA_1_eta0*modA_4_eta1);
  REAL eval_eta0_eta1_34(dof_34*modA_2_eta0*modA_4_eta1);
  REAL eval_eta0_eta1_35(dof_35*modA_3_eta0*modA_4_eta1);
  REAL eval_eta0_eta1_36(dof_36*modA_4_eta0*modA_4_eta1);
  REAL eval_eta0_eta1_37(dof_37*modA_4_eta1*modA_5_eta0);
  REAL eval_eta0_eta1_38(dof_38*modA_4_eta1*modA_6_eta0);
  REAL eval_eta0_eta1_39(dof_39*modA_4_eta1*modA_7_eta0);
  REAL eval_eta0_eta1_40(dof_40*modA_0_eta0*modA_5_eta1);
  REAL eval_eta0_eta1_41(dof_41*modA_1_eta0*modA_5_eta1);
  REAL eval_eta0_eta1_42(dof_42*modA_2_eta0*modA_5_eta1);
  REAL eval_eta0_eta1_43(dof_43*modA_3_eta0*modA_5_eta1);
  REAL eval_eta0_eta1_44(dof_44*modA_4_eta0*modA_5_eta1);
  REAL eval_eta0_eta1_45(dof_45*modA_5_eta0*modA_5_eta1);
  REAL eval_eta0_eta1_46(dof_46*modA_5_eta1*modA_6_eta0);
  REAL eval_eta0_eta1_47(dof_47*modA_5_eta1*modA_7_eta0);
  REAL eval_eta0_eta1_48(dof_48*modA_0_eta0*modA_6_eta1);
  REAL eval_eta0_eta1_49(dof_49*modA_1_eta0*modA_6_eta1);
  REAL eval_eta0_eta1_50(dof_50*modA_2_eta0*modA_6_eta1);
  REAL eval_eta0_eta1_51(dof_51*modA_3_eta0*modA_6_eta1);
  REAL eval_eta0_eta1_52(dof_52*modA_4_eta0*modA_6_eta1);
  REAL eval_eta0_eta1_53(dof_53*modA_5_eta0*modA_6_eta1);
  REAL eval_eta0_eta1_54(dof_54*modA_6_eta0*modA_6_eta1);
  REAL eval_eta0_eta1_55(dof_55*modA_6_eta1*modA_7_eta0);
  REAL eval_eta0_eta1_56(dof_56*modA_0_eta0*modA_7_eta1);
  REAL eval_eta0_eta1_57(dof_57*modA_1_eta0*modA_7_eta1);
  REAL eval_eta0_eta1_58(dof_58*modA_2_eta0*modA_7_eta1);
  REAL eval_eta0_eta1_59(dof_59*modA_3_eta0*modA_7_eta1);
  REAL eval_eta0_eta1_60(dof_60*modA_4_eta0*modA_7_eta1);
  REAL eval_eta0_eta1_61(dof_61*modA_5_eta0*modA_7_eta1);
  REAL eval_eta0_eta1_62(dof_62*modA_6_eta0*modA_7_eta1);
  REAL eval_eta0_eta1_63(dof_63*modA_7_eta0*modA_7_eta1);
  // 63
  REAL eval_eta0_eta1(eval_eta0_eta1_0 + eval_eta0_eta1_1 + eval_eta0_eta1_10 + eval_eta0_eta1_11 + eval_eta0_eta1_12 + eval_eta0_eta1_13 + eval_eta0_eta1_14 + eval_eta0_eta1_15 + eval_eta0_eta1_16 + eval_eta0_eta1_17 + eval_eta0_eta1_18 + eval_eta0_eta1_19 + eval_eta0_eta1_2 + eval_eta0_eta1_20 + eval_eta0_eta1_21 + eval_eta0_eta1_22 + eval_eta0_eta1_23 + eval_eta0_eta1_24 + eval_eta0_eta1_25 + eval_eta0_eta1_26 + eval_eta0_eta1_27 + eval_eta0_eta1_28 + eval_eta0_eta1_29 + eval_eta0_eta1_3 + eval_eta0_eta1_30 + eval_eta0_eta1_31 + eval_eta0_eta1_32 + eval_eta0_eta1_33 + eval_eta0_eta1_34 + eval_eta0_eta1_35 + eval_eta0_eta1_36 + eval_eta0_eta1_37 + eval_eta0_eta1_38 + eval_eta0_eta1_39 + eval_eta0_eta1_4 + eval_eta0_eta1_40 + eval_eta0_eta1_41 + eval_eta0_eta1_42 + eval_eta0_eta1_43 + eval_eta0_eta1_44 + eval_eta0_eta1_45 + eval_eta0_eta1_46 + eval_eta0_eta1_47 + eval_eta0_eta1_48 + eval_eta0_eta1_49 + eval_eta0_eta1_5 + eval_eta0_eta1_50 + eval_eta0_eta1_51 + eval_eta0_eta1_52 + eval_eta0_eta1_53 + eval_eta0_eta1_54 + eval_eta0_eta1_55 + eval_eta0_eta1_56 + eval_eta0_eta1_57 + eval_eta0_eta1_58 + eval_eta0_eta1_59 + eval_eta0_eta1_6 + eval_eta0_eta1_60 + eval_eta0_eta1_61 + eval_eta0_eta1_62 + eval_eta0_eta1_63 + eval_eta0_eta1_7 + eval_eta0_eta1_8 + eval_eta0_eta1_9);
  return eval_eta0_eta1;
}


/**
 * Class to evaluate Nektar++ fields by evaluating basis functions.
 */
template <typename T>
class FunctionEvaluateBasis : public BasisEvaluateBase<T> {
protected:
  /**
   *  Templated evaluation function for CRTP.
   */
  template <typename EVALUATE_TYPE, typename COMPONENT_TYPE>
  inline sycl::event evaluate_inner(
      ExpansionLooping::JacobiExpansionLoopingInterface<EVALUATE_TYPE>
          evaluation_type,
      ParticleGroupSharedPtr particle_group, Sym<COMPONENT_TYPE> sym,
      const int component) {

    const ShapeType shape_type = evaluation_type.get_shape_type();
    const int cells_iterset_size = this->map_shape_to_count.at(shape_type);
    if (cells_iterset_size == 0) {
      return sycl::event{};
    }

    const auto k_cells_iterset =
        this->map_shape_to_dh_cells.at(shape_type)->d_buffer.ptr;

    auto mpi_rank_dat = particle_group->mpi_rank_dat;

    const auto k_ref_positions =
        (*particle_group)[Sym<REAL>("NESO_REFERENCE_POSITIONS")]
            ->cell_dat.device_ptr();

    auto k_output = (*particle_group)[sym]->cell_dat.device_ptr();
    const int k_component = component;

    const auto d_npart_cell = mpi_rank_dat->d_npart_cell;
    const auto k_global_coeffs = this->dh_global_coeffs.d_buffer.ptr;
    const auto k_coeffs_offsets = this->dh_coeffs_offsets.d_buffer.ptr;
    const auto k_nummodes = this->dh_nummodes.d_buffer.ptr;

    // jacobi coefficients
    const auto k_coeffs_pnm10 = this->dh_coeffs_pnm10.d_buffer.ptr;
    const auto k_coeffs_pnm11 = this->dh_coeffs_pnm11.d_buffer.ptr;
    const auto k_coeffs_pnm2 = this->dh_coeffs_pnm2.d_buffer.ptr;
    const int k_stride_n = this->stride_n;

    const int k_max_total_nummodes0 =
        this->map_total_nummodes.at(shape_type).at(0);
    const int k_max_total_nummodes1 =
        this->map_total_nummodes.at(shape_type).at(1);
    const int k_max_total_nummodes2 =
        this->map_total_nummodes.at(shape_type).at(2);

    const size_t local_size = get_num_local_work_items(
        this->sycl_target,
        static_cast<size_t>(k_max_total_nummodes0 + k_max_total_nummodes1 +
                            k_max_total_nummodes2) *
            sizeof(REAL),
        128);

    const int local_mem_num_items =
        (k_max_total_nummodes0 + k_max_total_nummodes1 +
         k_max_total_nummodes2) *
        local_size;
    const size_t outer_size =
        get_particle_loop_global_size(mpi_rank_dat, local_size);

    const int k_ndim = evaluation_type.get_ndim();

    sycl::range<2> cell_iterset_range{static_cast<size_t>(cells_iterset_size),
                                      static_cast<size_t>(outer_size) *
                                          static_cast<size_t>(local_size)};
    sycl::range<2> local_iterset{1, local_size};

    auto event_loop = this->sycl_target->queue.submit([&](sycl::handler &cgh) {
      sycl::accessor<REAL, 1, sycl::access::mode::read_write,
                     sycl::access::target::local>
          local_mem(sycl::range<1>(local_mem_num_items), cgh);

      cgh.parallel_for<>(
          sycl::nd_range<2>(cell_iterset_range, local_iterset),
          [=](sycl::nd_item<2> idx) {
            const int iter_cell = idx.get_global_id(0);
            const int idx_local = idx.get_local_id(1);

            const INT cellx = k_cells_iterset[iter_cell];
            const INT layerx = idx.get_global_id(1);
            ExpansionLooping::JacobiExpansionLoopingInterface<EVALUATE_TYPE>
                loop_type{};

            if (layerx < d_npart_cell[cellx]) {
              const REAL *dofs = &k_global_coeffs[k_coeffs_offsets[cellx]];

              // Get the number of modes in x,y and z.
              const int nummodes = k_nummodes[cellx];

              REAL xi0, xi1, xi2, eta0, eta1, eta2;
              xi0 = k_ref_positions[cellx][0][layerx];
              if (k_ndim > 1) {
                xi1 = k_ref_positions[cellx][1][layerx];
              }
              if (k_ndim > 2) {
                xi2 = k_ref_positions[cellx][2][layerx];
              }

              loop_type.loc_coord_to_loc_collapsed(xi0, xi1, xi2, &eta0, &eta1,
                                                   &eta2);

              // Get the local space for the 1D evaluations in each dimension.
              REAL *local_space_0 =
                  &local_mem[idx_local *
                             (k_max_total_nummodes0 + k_max_total_nummodes1 +
                              k_max_total_nummodes2)];
              REAL *local_space_1 = local_space_0 + k_max_total_nummodes0;
              REAL *local_space_2 = local_space_1 + k_max_total_nummodes1;

              // Compute the basis functions in each dimension.
              loop_type.evaluate_basis_0(nummodes, eta0, k_stride_n,
                                         k_coeffs_pnm10, k_coeffs_pnm11,
                                         k_coeffs_pnm2, local_space_0);
              loop_type.evaluate_basis_1(nummodes, eta1, k_stride_n,
                                         k_coeffs_pnm10, k_coeffs_pnm11,
                                         k_coeffs_pnm2, local_space_1);
              loop_type.evaluate_basis_2(nummodes, eta2, k_stride_n,
                                         k_coeffs_pnm10, k_coeffs_pnm11,
                                         k_coeffs_pnm2, local_space_2);

              REAL evaluation = 0.0;
              loop_type.loop_evaluate(nummodes, dofs, local_space_0,
                                      local_space_1, local_space_2,
                                      &evaluation);

              k_output[cellx][k_component][layerx] = evaluation;
            }
          });
    });

    return event_loop;
  }


  /**
   *  Templated evaluation function for CRTP.
   */
  template <typename EVALUATE_TYPE, typename COMPONENT_TYPE>
  inline sycl::event evaluate_inner_per_cell(
      ExpansionLooping::JacobiExpansionLoopingInterface<EVALUATE_TYPE>
          evaluation_type,
      ParticleGroupSharedPtr particle_group, Sym<COMPONENT_TYPE> sym,
      const int component) {

    const ShapeType shape_type = evaluation_type.get_shape_type();
    const int cells_iterset_size = this->map_shape_to_count.at(shape_type);
    if (cells_iterset_size == 0) {
      return sycl::event{};
    }

    const auto k_cells_iterset =
        this->map_shape_to_dh_cells.at(shape_type)->d_buffer.ptr;

    auto mpi_rank_dat = particle_group->mpi_rank_dat;

    const auto k_ref_positions =
        (*particle_group)[Sym<REAL>("NESO_REFERENCE_POSITIONS")]
            ->cell_dat.device_ptr();

    auto k_output = (*particle_group)[sym]->cell_dat.device_ptr();
    const int k_component = component;

    const auto d_npart_cell = mpi_rank_dat->d_npart_cell;
    const auto k_global_coeffs = this->dh_global_coeffs.d_buffer.ptr;
    const auto k_coeffs_offsets = this->dh_coeffs_offsets.d_buffer.ptr;
    const auto k_nummodes = this->dh_nummodes.d_buffer.ptr;

    // jacobi coefficients
    const auto k_coeffs_pnm10 = this->dh_coeffs_pnm10.d_buffer.ptr;
    const auto k_coeffs_pnm11 = this->dh_coeffs_pnm11.d_buffer.ptr;
    const auto k_coeffs_pnm2 = this->dh_coeffs_pnm2.d_buffer.ptr;
    const int k_stride_n = this->stride_n;

    const int k_max_total_nummodes0 =
        this->map_total_nummodes.at(shape_type).at(0);
    const int k_max_total_nummodes1 =
        this->map_total_nummodes.at(shape_type).at(1);
    const int k_max_total_nummodes2 =
        this->map_total_nummodes.at(shape_type).at(2);

    const size_t local_size = get_num_local_work_items(
        this->sycl_target,
        static_cast<size_t>(k_max_total_nummodes0 + k_max_total_nummodes1 +
                            k_max_total_nummodes2) *
            sizeof(REAL),
        128);

    const int local_mem_num_items =
        (k_max_total_nummodes0 + k_max_total_nummodes1 +
         k_max_total_nummodes2) *
        local_size;
    const size_t outer_size =
        get_particle_loop_global_size(mpi_rank_dat, local_size);

    const int k_ndim = evaluation_type.get_ndim();


    EventStack event_stack;
    
    for(int cell_host=0 ; cell_host<cells_iterset_size ; cell_host++){
      const INT num_particles = mpi_rank_dat->h_npart_cell[cell_host];
      const INT k_cellx = cell_host;

      const auto div_mod =
          std::div(static_cast<long long>(num_particles), static_cast<long long>(local_size));
      const std::size_t outer_size =
          static_cast<std::size_t>(div_mod.quot + (div_mod.rem == 0 ? 0 : 1));

      sycl::range<1> cell_iterset_range{static_cast<size_t>(outer_size) *
                                          static_cast<size_t>(local_size)};
      sycl::range<1> local_iterset{local_size};

      auto event_loop = this->sycl_target->queue.submit([&](sycl::handler &cgh) {
        sycl::accessor<REAL, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            local_mem(sycl::range<1>(local_mem_num_items), cgh);

        cgh.parallel_for<>(
            sycl::nd_range<1>(cell_iterset_range, local_iterset),
            [=](sycl::nd_item<1> idx) {
              const int idx_local = idx.get_local_id(0);

              const INT cellx = k_cells_iterset[k_cellx];
              const INT layerx = idx.get_global_id(0);

              if (layerx < d_npart_cell[cellx]) {

                ExpansionLooping::JacobiExpansionLoopingInterface<EVALUATE_TYPE>
                    loop_type{};

                const REAL *dofs = &k_global_coeffs[k_coeffs_offsets[cellx]];

                // Get the number of modes in x,y and z.
                const int nummodes = k_nummodes[cellx];

                REAL xi0, xi1, xi2, eta0, eta1, eta2;
                xi0 = k_ref_positions[cellx][0][layerx];
                if (k_ndim > 1) {
                  xi1 = k_ref_positions[cellx][1][layerx];
                }
                if (k_ndim > 2) {
                  xi2 = k_ref_positions[cellx][2][layerx];
                }

                loop_type.loc_coord_to_loc_collapsed(xi0, xi1, xi2, &eta0, &eta1,
                                                     &eta2);

                // Get the local space for the 1D evaluations in each dimension.
                REAL *local_space_0 =
                    &local_mem[idx_local *
                               (k_max_total_nummodes0 + k_max_total_nummodes1 +
                                k_max_total_nummodes2)];
                REAL *local_space_1 = local_space_0 + k_max_total_nummodes0;
                REAL *local_space_2 = local_space_1 + k_max_total_nummodes1;

                // Compute the basis functions in each dimension.
                loop_type.evaluate_basis_0(nummodes, eta0, k_stride_n,
                                           k_coeffs_pnm10, k_coeffs_pnm11,
                                           k_coeffs_pnm2, local_space_0);
                loop_type.evaluate_basis_1(nummodes, eta1, k_stride_n,
                                           k_coeffs_pnm10, k_coeffs_pnm11,
                                           k_coeffs_pnm2, local_space_1);
                loop_type.evaluate_basis_2(nummodes, eta2, k_stride_n,
                                           k_coeffs_pnm10, k_coeffs_pnm11,
                                           k_coeffs_pnm2, local_space_2);

                REAL evaluation = 0.0;
                loop_type.loop_evaluate(nummodes, dofs, local_space_0,
                                        local_space_1, local_space_2,
                                        &evaluation);

                k_output[cellx][k_component][layerx] = evaluation;
              }
            });
      });
      event_stack.push(event_loop);
    }
    event_stack.wait();

    return sycl::event{};
  }

public:
  /// Disable (implicit) copies.
  FunctionEvaluateBasis(const FunctionEvaluateBasis &st) = delete;
  /// Disable (implicit) copies.
  FunctionEvaluateBasis &operator=(FunctionEvaluateBasis const &a) = delete;

  /**
   * Constructor to create instance to evaluate Nektar++ fields.
   *
   * @param field Example Nektar++ field of the same mesh and function space as
   * the destination fields that this instance will be called with.
   * @param mesh ParticleMeshInterface constructed over same mesh as the
   * function.
   * @param cell_id_translation Map between NESO-Particles cells and Nektar++
   * cells.
   */
  FunctionEvaluateBasis(std::shared_ptr<T> field,
                        ParticleMeshInterfaceSharedPtr mesh,
                        CellIDTranslationSharedPtr cell_id_translation)
      : BasisEvaluateBase<T>(field, mesh, cell_id_translation) {}

  /**
   * Evaluate nektar++ function at particle locations.
   *
   * @param particle_group Source container of particles.
   * @param sym Symbol of ParticleDat within the ParticleGroup.
   * @param component Determine which component of the ParticleDat is
   * the output for function evaluations.
   * @param global_coeffs source DOFs which are evaluated.
   */
  template <typename U, typename V>
  inline void evaluate(ParticleGroupSharedPtr particle_group, Sym<U> sym,
                       const int component, V &global_coeffs) {
    const int num_global_coeffs = global_coeffs.size();
    this->dh_global_coeffs.realloc_no_copy(num_global_coeffs);
    for (int px = 0; px < num_global_coeffs; px++) {
      this->dh_global_coeffs.h_buffer.ptr[px] = global_coeffs[px];
    }
    this->dh_global_coeffs.host_to_device();

    EventStack event_stack{};

/*
    if (this->mesh->get_ndim() == 2) {
      event_stack.push(evaluate_inner_per_cell(ExpansionLooping::Quadrilateral{},
                                      particle_group, sym, component));

      event_stack.push(evaluate_inner_per_cell(ExpansionLooping::Triangle{},
                                      particle_group, sym, component));
    } else {
      event_stack.push(evaluate_inner_per_cell(ExpansionLooping::Hexahedron{},
                                      particle_group, sym, component));
      event_stack.push(evaluate_inner_per_cell(ExpansionLooping::Pyramid{},
                                      particle_group, sym, component));
      event_stack.push(evaluate_inner_per_cell(ExpansionLooping::Prism{}, particle_group,
                                      sym, component));
      event_stack.push(evaluate_inner_per_cell(ExpansionLooping::Tetrahedron{},
                                      particle_group, sym, component));
    }
*/

    if (this->mesh->get_ndim() == 2) {
      event_stack.push(evaluate_inner(ExpansionLooping::Quadrilateral{},
                                      particle_group, sym, component));

      event_stack.push(evaluate_inner(ExpansionLooping::Triangle{},
                                      particle_group, sym, component));
    } else {
      event_stack.push(evaluate_inner(ExpansionLooping::Hexahedron{},
                                      particle_group, sym, component));
      event_stack.push(evaluate_inner(ExpansionLooping::Pyramid{},
                                      particle_group, sym, component));
      event_stack.push(evaluate_inner(ExpansionLooping::Prism{}, particle_group,
                                      sym, component));
      event_stack.push(evaluate_inner(ExpansionLooping::Tetrahedron{},
                                      particle_group, sym, component));
    }

    event_stack.wait();
  }


  template <size_t NUM_MODES, typename EVALUATE_TYPE, typename COMPONENT_TYPE>
  inline sycl::event evaluate_inner_test(
      ExpansionLooping::JacobiExpansionLoopingInterface<EVALUATE_TYPE>
          evaluation_type,
      ParticleGroupSharedPtr particle_group, Sym<COMPONENT_TYPE> sym,
      const int component) {

    const ShapeType shape_type = evaluation_type.get_shape_type();
    const int cells_iterset_size = this->map_shape_to_count.at(shape_type);
    if (cells_iterset_size == 0) {
      return sycl::event{};
    }

    const auto k_cells_iterset =
        this->map_shape_to_dh_cells.at(shape_type)->d_buffer.ptr;

    auto mpi_rank_dat = particle_group->mpi_rank_dat;

    const auto k_ref_positions =
        (*particle_group)[Sym<REAL>("NESO_REFERENCE_POSITIONS")]
            ->cell_dat.device_ptr();

    auto k_output = (*particle_group)[sym]->cell_dat.device_ptr();
    const int k_component = component;

    const auto d_npart_cell = mpi_rank_dat->d_npart_cell;
    const auto k_global_coeffs = this->dh_global_coeffs.d_buffer.ptr;
    const auto k_coeffs_offsets = this->dh_coeffs_offsets.d_buffer.ptr;
    const auto k_nummodes = this->dh_nummodes.d_buffer.ptr;

    // jacobi coefficients
    const auto k_coeffs_pnm10 = this->dh_coeffs_pnm10.d_buffer.ptr;
    const auto k_coeffs_pnm11 = this->dh_coeffs_pnm11.d_buffer.ptr;
    const auto k_coeffs_pnm2 = this->dh_coeffs_pnm2.d_buffer.ptr;
    const int k_stride_n = this->stride_n;

    const int k_max_total_nummodes0 =
        this->map_total_nummodes.at(shape_type).at(0);
    const int k_max_total_nummodes1 =
        this->map_total_nummodes.at(shape_type).at(1);
    const int k_max_total_nummodes2 =
        this->map_total_nummodes.at(shape_type).at(2);
/*
    const size_t local_size = get_num_local_work_items(
        this->sycl_target,
        static_cast<size_t>(k_max_total_nummodes0 + k_max_total_nummodes1 +
                            k_max_total_nummodes2) *
            sizeof(REAL),
        128);
*/
    const size_t local_size = 128;
    const int local_mem_num_items =
        (k_max_total_nummodes0 + k_max_total_nummodes1 +
         k_max_total_nummodes2) *
        local_size;
    const size_t outer_size =
        get_particle_loop_global_size(mpi_rank_dat, local_size);

    const int k_ndim = evaluation_type.get_ndim();

    sycl::range<2> cell_iterset_range{static_cast<size_t>(cells_iterset_size),
                                      static_cast<size_t>(outer_size) *
                                          static_cast<size_t>(local_size)};
    sycl::range<2> local_iterset{1, local_size};

    auto event_loop = this->sycl_target->queue.submit([&](sycl::handler &cgh) {
      //sycl::accessor<REAL, 1, sycl::access::mode::read_write,
      //               sycl::access::target::local>
      //    local_mem(sycl::range<1>(local_mem_num_items), cgh);

      cgh.parallel_for<>(
          sycl::nd_range<2>(cell_iterset_range, local_iterset),
          [=](sycl::nd_item<2> idx) {
            const int iter_cell = idx.get_global_id(0);
            const int idx_local = idx.get_local_id(1);

            const INT cellx = k_cells_iterset[iter_cell];
            const INT layerx = idx.get_global_id(1);
            //ExpansionLooping::JacobiExpansionLoopingInterface<EVALUATE_TYPE>
            //    loop_type{};

            if (layerx < d_npart_cell[cellx]) {
              const REAL *dofs = &k_global_coeffs[k_coeffs_offsets[cellx]];

              // Get the number of modes in x,y and z.
              const int nummodes = k_nummodes[cellx];

              REAL xi0, xi1, xi2, eta0, eta1, eta2;
              //xi0 = k_ref_positions[cellx][0][layerx];
              //if (k_ndim > 1) {
              //  xi1 = k_ref_positions[cellx][1][layerx];
              //}
              //if (k_ndim > 2) {
              //  xi2 = k_ref_positions[cellx][2][layerx];
              //}

              xi0 = k_ref_positions[cellx][0][layerx];
              xi1 = k_ref_positions[cellx][1][layerx];

              //loop_type.loc_coord_to_loc_collapsed(xi0, xi1, xi2, &eta0, &eta1,
              //                                     &eta2);
              eta0 = xi0;
              eta1 = xi1;
/*
              // Get the local space for the 1D evaluations in each dimension.
              REAL *local_space_0 =
                  &local_mem[idx_local *
                             (k_max_total_nummodes0 + k_max_total_nummodes1 +
                              k_max_total_nummodes2)];
              REAL *local_space_1 = local_space_0 + k_max_total_nummodes0;
              REAL *local_space_2 = local_space_1 + k_max_total_nummodes1;

              // Compute the basis functions in each dimension.
              loop_type.evaluate_basis_0(nummodes, eta0, k_stride_n,
                                         k_coeffs_pnm10, k_coeffs_pnm11,
                                         k_coeffs_pnm2, local_space_0);
              loop_type.evaluate_basis_1(nummodes, eta1, k_stride_n,
                                         k_coeffs_pnm10, k_coeffs_pnm11,
                                         k_coeffs_pnm2, local_space_1);
              loop_type.evaluate_basis_2(nummodes, eta2, k_stride_n,
                                         k_coeffs_pnm10, k_coeffs_pnm11,
                                         k_coeffs_pnm2, local_space_2);

              REAL evaluation = 0.0;
              loop_type.loop_evaluate(nummodes, dofs, local_space_0,
                                      local_space_1, local_space_2,
                                      &evaluation);
*/

              const REAL evaluation = quadrilateral_evaluate_scalar<NUM_MODES>(eta0, eta1, dofs);

              k_output[cellx][k_component][layerx] = evaluation;
            }
          });
    });

    return event_loop;
  }

  template <size_t NUM_MODES, typename EVALUATE_TYPE, typename COMPONENT_TYPE>
  inline sycl::event evaluate_inner_per_cell_test(
      ExpansionLooping::JacobiExpansionLoopingInterface<EVALUATE_TYPE>
          evaluation_type,
      ParticleGroupSharedPtr particle_group, Sym<COMPONENT_TYPE> sym,
      const int component) {

    const ShapeType shape_type = evaluation_type.get_shape_type();
    const int cells_iterset_size = this->map_shape_to_count.at(shape_type);
    if (cells_iterset_size == 0) {
      return sycl::event{};
    }

    const auto k_cells_iterset =
        this->map_shape_to_dh_cells.at(shape_type)->d_buffer.ptr;

    auto mpi_rank_dat = particle_group->mpi_rank_dat;

    const auto k_ref_positions =
        (*particle_group)[Sym<REAL>("NESO_REFERENCE_POSITIONS")]
            ->cell_dat.device_ptr();

    auto k_output = (*particle_group)[sym]->cell_dat.device_ptr();
    const int k_component = component;

    const auto d_npart_cell = mpi_rank_dat->d_npart_cell;
    const auto k_global_coeffs = this->dh_global_coeffs.d_buffer.ptr;
    const auto k_coeffs_offsets = this->dh_coeffs_offsets.d_buffer.ptr;
    const auto k_nummodes = this->dh_nummodes.d_buffer.ptr;

    // jacobi coefficients
    const auto k_coeffs_pnm10 = this->dh_coeffs_pnm10.d_buffer.ptr;
    const auto k_coeffs_pnm11 = this->dh_coeffs_pnm11.d_buffer.ptr;
    const auto k_coeffs_pnm2 = this->dh_coeffs_pnm2.d_buffer.ptr;
    const int k_stride_n = this->stride_n;

    const int k_max_total_nummodes0 =
        this->map_total_nummodes.at(shape_type).at(0);
    const int k_max_total_nummodes1 =
        this->map_total_nummodes.at(shape_type).at(1);
    const int k_max_total_nummodes2 =
        this->map_total_nummodes.at(shape_type).at(2);
/*
    const size_t local_size = get_num_local_work_items(
        this->sycl_target,
        static_cast<size_t>(k_max_total_nummodes0 + k_max_total_nummodes1 +
                            k_max_total_nummodes2) *
            sizeof(REAL),
        128);
*/
    const size_t local_size = 128;
    const int local_mem_num_items =
        (k_max_total_nummodes0 + k_max_total_nummodes1 +
         k_max_total_nummodes2) *
        local_size;
    const size_t outer_size =
        get_particle_loop_global_size(mpi_rank_dat, local_size);

    const int k_ndim = evaluation_type.get_ndim();

    sycl::range<2> cell_iterset_range{static_cast<size_t>(cells_iterset_size),
                                      static_cast<size_t>(outer_size) *
                                          static_cast<size_t>(local_size)};
    sycl::range<2> local_iterset{1, local_size};


    EventStack event_stack{};

    for(int cellx=0 ; cellx<cells_iterset_size ; cellx++){

    auto event_loop = this->sycl_target->queue.submit([&](sycl::handler &cgh) {
      //sycl::accessor<REAL, 1, sycl::access::mode::read_write,
      //               sycl::access::target::local>
      //    local_mem(sycl::range<1>(local_mem_num_items), cgh);
      const int num_particles = mpi_rank_dat->h_npart_cell[cellx]; 

      cgh.parallel_for<>(
          sycl::range<1>(static_cast<size_t>(num_particles)),
          [=](sycl::id<1> idx) {
            //const int iter_cell = idx.get_global_id(0);
            //const int idx_local = idx.get_local_id(1);

            //const INT cellx = k_cells_iterset[iter_cell];
            const INT layerx = idx;
            //ExpansionLooping::JacobiExpansionLoopingInterface<EVALUATE_TYPE>
            //    loop_type{};

            //if (layerx < d_npart_cell[cellx]) {
              const REAL *dofs = &k_global_coeffs[k_coeffs_offsets[cellx]];

              // Get the number of modes in x,y and z.
              const int nummodes = k_nummodes[cellx];

              REAL xi0, xi1, xi2, eta0, eta1, eta2;
              //xi0 = k_ref_positions[cellx][0][layerx];
              //if (k_ndim > 1) {
              //  xi1 = k_ref_positions[cellx][1][layerx];
              //}
              //if (k_ndim > 2) {
              //  xi2 = k_ref_positions[cellx][2][layerx];
              //}

              xi0 = k_ref_positions[cellx][0][layerx];
              xi1 = k_ref_positions[cellx][1][layerx];

              //loop_type.loc_coord_to_loc_collapsed(xi0, xi1, xi2, &eta0, &eta1,
              //                                     &eta2);
              eta0 = xi0;
              eta1 = xi1;
/*
              // Get the local space for the 1D evaluations in each dimension.
              REAL *local_space_0 =
                  &local_mem[idx_local *
                             (k_max_total_nummodes0 + k_max_total_nummodes1 +
                              k_max_total_nummodes2)];
              REAL *local_space_1 = local_space_0 + k_max_total_nummodes0;
              REAL *local_space_2 = local_space_1 + k_max_total_nummodes1;

              // Compute the basis functions in each dimension.
              loop_type.evaluate_basis_0(nummodes, eta0, k_stride_n,
                                         k_coeffs_pnm10, k_coeffs_pnm11,
                                         k_coeffs_pnm2, local_space_0);
              loop_type.evaluate_basis_1(nummodes, eta1, k_stride_n,
                                         k_coeffs_pnm10, k_coeffs_pnm11,
                                         k_coeffs_pnm2, local_space_1);
              loop_type.evaluate_basis_2(nummodes, eta2, k_stride_n,
                                         k_coeffs_pnm10, k_coeffs_pnm11,
                                         k_coeffs_pnm2, local_space_2);

              REAL evaluation = 0.0;
              loop_type.loop_evaluate(nummodes, dofs, local_space_0,
                                      local_space_1, local_space_2,
                                      &evaluation);
*/

              const REAL evaluation = quadrilateral_evaluate_scalar<NUM_MODES>(eta0, eta1, dofs);

              k_output[cellx][k_component][layerx] = evaluation;
            //}
          });
    });

      event_stack.push(event_loop);

    }

    event_stack.wait();
    
    return sycl::event{};
  }


  template <size_t NUM_MODES, typename EVALUATE_TYPE, typename COMPONENT_TYPE>
  inline sycl::event evaluate_inner_per_cell_vector_test(
      ExpansionLooping::JacobiExpansionLoopingInterface<EVALUATE_TYPE>
          evaluation_type,
      ParticleGroupSharedPtr particle_group, Sym<COMPONENT_TYPE> sym,
      const int component) {

    const ShapeType shape_type = evaluation_type.get_shape_type();
    const int cells_iterset_size = this->map_shape_to_count.at(shape_type);
    if (cells_iterset_size == 0) {
      return sycl::event{};
    }

    const auto k_cells_iterset =
        this->map_shape_to_dh_cells.at(shape_type)->d_buffer.ptr;

    auto mpi_rank_dat = particle_group->mpi_rank_dat;

    const auto k_ref_positions =
        (*particle_group)[Sym<REAL>("NESO_REFERENCE_POSITIONS")]
            ->cell_dat.device_ptr();

    auto k_output = (*particle_group)[sym]->cell_dat.device_ptr();
    const int k_component = component;

    const auto d_npart_cell = mpi_rank_dat->d_npart_cell;
    const auto k_global_coeffs = this->dh_global_coeffs.d_buffer.ptr;
    const auto k_coeffs_offsets = this->dh_coeffs_offsets.d_buffer.ptr;
    const auto k_nummodes = this->dh_nummodes.d_buffer.ptr;

    // jacobi coefficients
    const auto k_coeffs_pnm10 = this->dh_coeffs_pnm10.d_buffer.ptr;
    const auto k_coeffs_pnm11 = this->dh_coeffs_pnm11.d_buffer.ptr;
    const auto k_coeffs_pnm2 = this->dh_coeffs_pnm2.d_buffer.ptr;
    const int k_stride_n = this->stride_n;

    const int k_max_total_nummodes0 =
        this->map_total_nummodes.at(shape_type).at(0);
    const int k_max_total_nummodes1 =
        this->map_total_nummodes.at(shape_type).at(1);
    const int k_max_total_nummodes2 =
        this->map_total_nummodes.at(shape_type).at(2);
/*
    const size_t local_size = get_num_local_work_items(
        this->sycl_target,
        static_cast<size_t>(k_max_total_nummodes0 + k_max_total_nummodes1 +
                            k_max_total_nummodes2) *
            sizeof(REAL),
        128);
*/
    const size_t local_size = 128;
    const int local_mem_num_items =
        (k_max_total_nummodes0 + k_max_total_nummodes1 +
         k_max_total_nummodes2) *
        local_size;
    const size_t outer_size =
        get_particle_loop_global_size(mpi_rank_dat, local_size);

    const int k_ndim = evaluation_type.get_ndim();

    sycl::range<2> cell_iterset_range{static_cast<size_t>(cells_iterset_size),
                                      static_cast<size_t>(outer_size) *
                                          static_cast<size_t>(local_size)};
    sycl::range<2> local_iterset{1, local_size};


    EventStack event_stack{};

    for(int cellx=0 ; cellx<cells_iterset_size ; cellx++){

    auto event_loop = this->sycl_target->queue.submit([&](sycl::handler &cgh) {
      //sycl::accessor<REAL, 1, sycl::access::mode::read_write,
      //               sycl::access::target::local>
      //    local_mem(sycl::range<1>(local_mem_num_items), cgh);
      const int num_particles = mpi_rank_dat->h_npart_cell[cellx]; 

      const auto div_mod =
          std::div(static_cast<long long>(num_particles), static_cast<long long>(VECTOR_LENGTH));
      const std::size_t num_blocks =
          static_cast<std::size_t>(div_mod.quot + (div_mod.rem == 0 ? 0 : 1));

      cgh.parallel_for<>(
          sycl::range<1>(static_cast<size_t>(num_blocks)),
          [=](sycl::id<1> idx) {

            const REAL *dofs = &k_global_coeffs[k_coeffs_offsets[cellx]];

            const INT layer_start = idx * VECTOR_LENGTH;
            const INT layer_end = std::min(INT(layer_start + VECTOR_LENGTH), INT(num_particles));

            REAL xi0_local[VECTOR_LENGTH];
            REAL xi1_local[VECTOR_LENGTH];
            REAL eval_local[VECTOR_LENGTH];
            for(int ix=0 ; ix<VECTOR_LENGTH ; ix++){
              xi0_local[ix] = 0.0;
              xi1_local[ix] = 0.0;
              eval_local[ix] = 0.0;
            }
            int cx = 0;
            for(int ix=layer_start ; ix<layer_end ; ix++){
              xi0_local[cx] = k_ref_positions[cellx][0][ix];
              xi1_local[cx] = k_ref_positions[cellx][1][ix];
              cx++;
            }

            sycl::local_ptr<const REAL> xi0_ptr(xi0_local);
            sycl::local_ptr<const REAL> xi1_ptr(xi1_local);

            sycl::vec<REAL, VECTOR_LENGTH> xi0, xi1;
            xi0.load(0, xi0_ptr);
            xi1.load(0, xi1_ptr);

            const sycl::vec<REAL, VECTOR_LENGTH> eval = quadrilateral_evaluate_vector<NUM_MODES>(xi0, xi1, dofs);

            sycl::local_ptr<REAL> eval_ptr(eval_local);
            eval.store(0, eval_ptr);

            cx = 0;
            for(int ix=layer_start ; ix<layer_end ; ix++){
              k_output[cellx][k_component][ix] = eval_local[cx];
              cx++;
            }
          });
    });

      event_stack.push(event_loop);

    }

    event_stack.wait();
    
    return sycl::event{};
  }




  template <typename U, typename V>
  inline void evaluate_test_init(ParticleGroupSharedPtr particle_group, Sym<U> sym,
                       const int component, V &global_coeffs) {
    const int num_global_coeffs = global_coeffs.size();
    this->dh_global_coeffs.realloc_no_copy(num_global_coeffs);
    for (int px = 0; px < num_global_coeffs; px++) {
      this->dh_global_coeffs.h_buffer.ptr[px] = global_coeffs[px];
    }
    this->dh_global_coeffs.host_to_device();
  }

  template <typename U, typename V>
  void evaluate_test(ParticleGroupSharedPtr particle_group, Sym<U> sym,
                       const int component, V &global_coeffs) {

    const auto num_modes = this->dh_nummodes.h_buffer.ptr[0];

    EventStack event_stack{};
    //event_stack.push(evaluate_inner_per_cell(ExpansionLooping::Quadrilateral{},
    //                                         particle_group, sym, component));

    /*
    if (num_modes == 4){
      event_stack.push(evaluate_inner_per_cell_test<4>(ExpansionLooping::Quadrilateral{},
                                              particle_group, sym, component));
    } else if (num_modes == 8){
      event_stack.push(evaluate_inner_per_cell_test<8>(ExpansionLooping::Quadrilateral{},
                                              particle_group, sym, component));
    }
    */

    if (num_modes == 4){
      event_stack.push(evaluate_inner_per_cell_vector_test<4>(ExpansionLooping::Quadrilateral{},
                                              particle_group, sym, component));
    } else if (num_modes == 8){
      event_stack.push(evaluate_inner_per_cell_vector_test<8>(ExpansionLooping::Quadrilateral{},
                                              particle_group, sym, component));
    }

    event_stack.wait();

  }
};

} // namespace NESO

#endif
