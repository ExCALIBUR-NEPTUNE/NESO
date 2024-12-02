#include "include/args.hpp"
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
static int get_int(char *str) {
  char *endptr = nullptr;
  auto n = strtol(str, &endptr, 10);
  if (n == LONG_MIN || n == LONG_MAX) {
    perror("Argument under/overflow");
  }
  if (endptr == str) {
    // NOTE: I think this is not safe using printf with user input
    // so don't use code for anything "real"
    fprintf(stderr, "%s is not valid value\n", str);
    exit(1);
  }
  if (n <= 0) {
    fprintf(stderr, "argument values must be positive\n");
    exit(1);
  }
  // TODO: should I just change to long everywhere?
  if (n > INT_MAX) {
    fprintf(stderr, "Max argument value is %d\n", INT_MAX);
    exit(1);
  }
  return (int)n;
}

void print_options(CmdArgs const &arg) {
  printf("Options:\n"
         "number of cells %d\n"
         "number of active cells (have particles) %d\n"
         "max particles per cell %d\n"
         "min particles per cell %d\n",
         arg.ncell, arg.active_cell, arg.max_per_cell, arg.min_per_cell);
}

void print_help() {
  printf("-n\tnumber of cells\n"
         "-a\tnumber of cells with particles\n"
         "-t\tmax number of particles in cell\n"
         "-l\tmin number of particles in cell\n"
         "-h\tprint this message\n"
         "--\tNEED THIS AS last option before any --benchmark_* "
         "options\n"); // hack to make it work with googlebenchmark
}

// Not using getopt in case someone want to use windows
// Probably not 100% robust to wrong input
CmdArgs get_args(int argc, char **argv, bool print) {
  CmdArgs arg;
  int count = 1;
  while (count < (argc - 1)) {
    if (argv[count][0] != '-')
      continue;
    switch (argv[count][1]) {
    case 'n':
      arg.ncell = get_int(argv[count + 1]);
      break;
    case 'a':
      arg.active_cell = get_int(argv[count + 1]);
      break;
    case 't':
      arg.max_per_cell = get_int(argv[count + 1]);
      break;
    case 'l':
      arg.min_per_cell = get_int(argv[count + 1]);
      break;
    case '-':
      goto END;
    default:
      printf("Unknown argument: -%c\n", argv[count][1]);
    case 'h':
      print_help();
      // options
      exit(0);
    }
    count += 2;
  }
END:
  if (arg.active_cell > arg.ncell) {
    fprintf(stderr,
            "Active cells (%d) is greater than number of cells (%d)\n"
            "Resetting active cells = ncell\n",
            arg.active_cell, arg.ncell);
    arg.active_cell = arg.ncell;
  }
  if (arg.min_per_cell > arg.max_per_cell) {
    fprintf(stderr,
            "Max particle per cells (%d) is greater than min particles per "
            "cell (%d)\n"
            "Resetting active min = max\n",
            arg.max_per_cell, arg.min_per_cell);
    arg.min_per_cell = arg.max_per_cell;
  }
  if (print)
    print_options(arg);
  return arg;
}
