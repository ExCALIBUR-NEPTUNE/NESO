#pragma once

// Just somewhere to lump constants while developing
// TODO think about this properly
namespace NESO::Project::Constants
{
constexpr int local_size = 128;
constexpr int private_mem_size = 20;
constexpr int vector_width = 4;
constexpr int beta = 1;
constexpr int alpha = 1;
constexpr int cpu_stride = 1;
constexpr int gpu_stride = local_size + 1;
} // namespace Constants
