# clang-format
find ./src ./include ./test ./solvers -iname \*.hpp -o -iname \*.cpp | xargs clang-format -i
# cmake-format
cmake-format -i CMakeLists.txt # Do this one on it's own so we don't go into build etc
find ./src ./include ./test ./solvers -iname CMakeLists.txt | xargs cmake-format -i
# black
find ./python -iname \*.py | xargs black

