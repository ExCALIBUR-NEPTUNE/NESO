// Gmsh project created on Thu Feb  9 10:29:04 2017
Point(1) = {0, 0, 0, 1.0};
Point(2) = {100, 0, 0, 1.0};
Point(3) = {100, 100, 0, 1.0};
Point(4) = {0, 100, 0, 1.0};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Transfinite Line {1,2,3,4} = 81;
Line Loop(5) = {4, 1, 2, 3};
Plane Surface(6) = {5};
Transfinite Surface {6} Alternated;
Recombine Surface {6};

Physical Line("1") = {4};
Physical Line("2") = {3};
Physical Line("3") = {2};
Physical Line("4") = {1};
Physical Surface("Regions") = {6};
