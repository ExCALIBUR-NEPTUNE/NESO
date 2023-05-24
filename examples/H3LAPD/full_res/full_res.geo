//=================== Parameter names, chosen to match H3 =====================
// Physical sizes / m
Rmax   = 0.4;
length = 17;
// # cells in each dimension
nx        = 85; // Radial resolution
ny        = 16; // Parallel (on-axis) resolution
nz_target = 64; // Target Azimuthal resolution
// N.B.
// nx = 64 for H3, but they use Rmin=0.1; match approximately by increasing to nint[64 * 0.4 / (0.4-0.1)]
// nz is only a target - the actual value used is 3*Floor(nz_target,3)
//=============================================================================

// Create a line in the x-direction of length <Rmax>, with <nx> divisions
Point(1) = {0, 0, 0, Rmax};
Point(2) = {1, 0, 0, Rmax};
Line(1) = {1, 2};
Transfinite Line(1) = nx+1;

// Extrude split line into meshed circle
// (Rotation extrusion can only cope with angles < pi, so has to be done in three parts...)
nz_over3 = Floor(nz_target/3);
Printf("Using nz = %g",3*nz_over3);
c1[] = Extrude {{0, 0, 1}, {0, 0, 0}, 2*Pi/3} {
  Curve{1}; Layers{nz_over3}; Recombine;
};

c2[] = Extrude {{0, 0, 1}, {0, 0, 0}, 2*Pi/3} {
  Curve{c1[0]}; Layers{nz_over3}; Recombine;
};

c3[] = Extrude {{0, 0, 1}, {0, 0, 0}, 2*Pi/3} {
  Curve{c2[0]}; Layers{nz_over3}; Recombine;
};

// Extrude each 1/3 of a circle to 1/3 of a cylinder
v1[] = Extrude {0,0,length} {Surface{c1[1]}; Layers{ny}; Recombine;};
v2[] = Extrude {0,0,length} {Surface{c2[1]}; Layers{ny}; Recombine;};
v3[] = Extrude {0,0,length} {Surface{c3[1]}; Layers{ny}; Recombine;};

// Label volumes and surfaces (combining the 3 azimuthally-split sections in each case)
// Whole domain volume
Physical Volume(0) = {v1[1],v2[1],v3[1]};
// Curved boundary surface
Physical Surface(1) = {v1[3],v2[3],v3[3]};
// Two circular boundary surfaces
Physical Surface(2) = {c1[1],c2[1],c3[1]};
Physical Surface(3) = {v1[0],v2[0],v3[0]};