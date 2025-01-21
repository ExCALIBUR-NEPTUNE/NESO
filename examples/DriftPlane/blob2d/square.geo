scale=1.0;

width=scale;
height=scale;
res=64/scale;  // number of elements per unit length

Point(1) = {-width/2,-height/2,0,0.1};
l[] = Extrude {width,0,0} {
      Point{1}; Layers{width*res}; Recombine;
};
s[] = Extrude {0,height,0} {
      Line{l[1]}; Layers{height*res}; Recombine;
};
Physical Surface(0) = {s[1]};
Physical Line(1) = {1};
Physical Line(2) = {3};
Physical Line(3) = {2};
Physical Line(4) = {4};
