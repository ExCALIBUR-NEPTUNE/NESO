using HDF5, Makie, CairoMakie, FFTW, LinearAlgebra, Statistics, LightXML

function foo()

xmlroot = root(parse_file(dirname(@__FILE__) * "/ebw_conditions.xml"));

parametersdict = Dict()
for i in child_nodes(xmlroot["CONDITIONS"][1]["PARAMETERS"][1])
  is_textnode(i) && continue
  try
    key, valuestring = strip.(split(strip(LightXML.content(XMLElement(i))), "="))
    try
      parametersdict[key] = parse(Int, valuestring)
    catch
      parametersdict[key] = parse(Float64, valuestring)
    end
  catch
  end
end

vth = parametersdict["particle_thermal_velocity"]
NT = parametersdict["particle_num_time_steps"]
NS = parametersdict["line_field_deriv_evaluations_step"]
dt = parametersdict["particle_time_step"]

ndiagx = parametersdict["line_field_deriv_evaluations_numx"]
ndiagy = parametersdict["line_field_deriv_evaluations_numy"]

NP = length(h5read("Electrostatic2D3V_particle_trajectory.h5part", "Step#0/V_0"))
Bx = h5read("Electrostatic2D3V_field_trajectory.h5", "global_data/B_x")[1]
By = h5read("Electrostatic2D3V_field_trajectory.h5", "global_data/B_y")[1]
Bz = h5read("Electrostatic2D3V_field_trajectory.h5", "global_data/B_z")[1]
Lx = h5read("Electrostatic2D3V_field_trajectory.h5", "global_data/L_x")[1]
Ly = h5read("Electrostatic2D3V_field_trajectory.h5", "global_data/L_y")[1]
weight = h5read("Electrostatic2D3V_field_trajectory.h5", "global_data/w")[1]
charge = h5read("Electrostatic2D3V_field_trajectory.h5", "global_data/q")[1]
mass = h5read("Electrostatic2D3V_field_trajectory.h5", "global_data/m")[1]
@show NP, Bx, By, Bz, Lx, Ly, weight, charge, mass, vth
@assert charge == mass == Lx == Ly == 1
volume = Lx * Ly

n0 = weight * NP / volume
B0 = sqrt(Bx^2 + By^2 + Bz^2)
Wc = charge * B0 / mass
Wp = sqrt(charge^2 * n0 / mass)

fname_phi = "Electrostatic2D3V_line_field_evaluations.h5part"
fname_Exy = "Electrostatic2D3V_line_field_deriv_evaluations.h5part"

function getindices(fname=fname_phi)
  x = h5read(fname, "Step#0/x")
  y = h5read(fname, "Step#0/y")
  @assert length(x) == length(y)
  x = h5read(fname, "Step#0/x");
  y = h5read(fname, "Step#0/y");

  xy = [i for i in eachrow(hcat(x, y))];

  perm = sortperm(xy, by=x->(x[1], x[2]));
  return perm
end

perminds = getindices()
NG = Int(sqrt(length(perminds)))

function get3D(fname, str)
  fz(i) = h5read(fname, "Step#$(i * NS)/" * str)

  output = cat([reshape(fz(i)[perminds], NG, NG) for i in 0:(NT÷NS)-1]..., dims=3);
  @assert size(output, 1) == ndiagx
  @assert size(output, 2) == ndiagy
  @assert size(output, 3) == NT÷NS
  return output
end

x2d = get3D(fname_phi, "x")
y2d = get3D(fname_phi, "y")

phis = get3D(fname_phi, "FIELD_EVALUATION_0")
Exs = get3D(fname_Exy, "FIELD_EVALUATION_0")
Eys = get3D(fname_Exy, "FIELD_EVALUATION_1")

NF = size(x2d, 2)
@show NG, NP, NP÷NG^2, NT, NS, NF, dt


Wc = abs(B0)
Wp = sqrt(n0)

function makespatiotemporalfigs(field, fieldstr, fnamestr)
  fig = Figure(; resolution=(600, 400))
  ax = Axis(fig[1, 1], xlabel=L"Space $[v_{th} / \Omega]$", ylabel=L"Time $[\tau_c]$")

  ts = (1:NF) .* (NS * dt) / (2pi / Wc)
  xs = (1:NG) .* Lx / NG / (vth / Wc)
  _, _, plotobj = heatmap!(ax, xs, ts, field)
  Colorbar(fig[1, 2], pltobj)
  save(fieldstr * "$fnamestr.png", fig)
end
function makespatialfigs(field, fieldstr, T)
  fig = Figure(; resolution=(600, 400))
  ax = Axis(fig[1, 1], xlabel=L"Space $[v_{th} / \Omega]$", ylabel=L"Time $[\tau_c]$")

  ts = (1:NF) .* (NS * dt) / (2pi / Wc)
  xs = (1:NG) .* Lx / NG / (vth / Wc)
  _, _, plotobj = heatmap!(ax, xs, ts, field)
  Colorbar(fig[1, 2], pltobj)
  save(fieldstr * "_XY_$T.png", fig)
end

function makefourierfigs(field, fieldstr)
  ws = ((1:NF) .* 2pi / (dt * NT) ./ Wc)
  wind = findlast(ws .< max(5.1, 1.1 * Wp / Wc));

  fig = Figure(; resolution=(600, 400))
  ax = Axis(fig[1, 1],
            xlabel=L"Wavenumber $[\Omega/v_{th}]$",
            ylabel=L"Frequency $[\Omega]$")
  Z = log10.(sum(i->abs.(fft(field[:, i, :])[2:end÷2-1, 1:wind]), 1:size(field, 2)))'
  ks = ((1:NG) .* 2pi / Lx .* vth / Wc)[2:end÷2]
  ws = ((1:NF) .* 2pi / (dt * NT) ./ Wc)[1:wind]
  _, _, plotobj = heatmap!(ax, ks, ws, Z)
  Colorbar(fig[1, 2], pltobj)
  save(fieldstr * "_WK_c.png", fig)

  fig = Figure(; resolution=(600, 400))
  ax = Axis(fig[1, 1],
            xlabel=L"Wavenumber $[\Pi/v_{th}]$",
            ylabel=L"Frequency $[\Pi]$")
  Z = log10.(sum(i->abs.(fft(field[:, i, :])[2:end÷2-1, 1:wind]), 1:size(field, 2)))'
  ks = ((1:NG) .* 2pi / Lx .* vth / Wp)[2:end÷2]
  ws = ((1:NF) .* 2pi / (dt * NT) ./ Wp)[1:wind]
  _, _, plotobj = heatmap!(ax, ks, ws, Z)
  Colorbar(fig[1, 2], pltobj)
  save(fieldstr * "_WK_p.png", fig)
end

for (field, fieldstr) in ((Exs, "Ex"), (Eys, "Ey"), (phis, "phi"))
  try
      makespatialfigs(field[:, :, 1], fieldstr, 1)
      makespatialfigs(field[:, :, 2], fieldstr, 2)
      makespatialfigs(field[:, :, end÷2], fieldstr, size(field, 3)÷2)
      makespatialfigs(field[:, :, end], fieldstr, size(field, 3))
  catch e
    @show e
    @warn "Failed to plot spatial plots"
  end

  try
      makespatiotemporalfigs(field[:, 1, :], fieldstr, "_XT_1")
      makespatiotemporalfigs(field[:, end, :], fieldstr, "_XT_$(size(field), 2)")
      makespatiotemporalfigs(field[1, :, :], fieldstr, "_YT_1")
      makespatiotemporalfigs(field[end, :, :], fieldstr, "_YT_$(size(field), 1)")
  catch e
    @show e
    @warn "Failed to plot spatio-temporal plots"
  end

  try
    makefourierfigs(field, fieldstr)
  catch e
    @show e
    @warn "Failed to plot Fourier plots"
  end
end

end

foo()
