#pragma once

struct CmdArgs {
  int ncell = 2024;
  int active_cell = 20;
  int max_per_cell = 2000;
  int min_per_cell = 600;
};

CmdArgs get_args(int argc, char **argv, bool print = false);
