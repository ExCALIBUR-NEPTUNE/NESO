#pragma once

// Just somewhere to lump constants while developing
// TODO think about this properly
namespace NESO::Project::Constants {
// Not all used.... yet
constexpr int preferred_block_size = 128;
constexpr int vector_width = 4;
constexpr int beta = 1;
constexpr int alpha = 1;
constexpr int cpu_stride = 1;
constexpr int gpu_warp_size = 32;
constexpr double Tolerance = 1.0E-12;
} // namespace NESO::Project::Constants
