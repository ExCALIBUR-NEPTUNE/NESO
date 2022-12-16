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

fname0 = "Electrostatic2D3V_line_field_evaluations.h5part"

function getindices(fname=fname0)
  x = h5read(fname0, "Step#0/x")
  y = h5read(fname0, "Step#0/y")
  @assert length(x) == length(y)

  indxs = findall(h5read(fname0, "Step#0/DIRECTION_0") .== 0)
  indys = findall(h5read(fname0, "Step#0/DIRECTION_0") .!= 0)
  indxs = sort(indxs, by=i->x[i])
  indys = sort(indys, by=i->y[i])

  @assert length(indxs) == length(indys)
  @assert length(indxs) == length(unique(indxs))
  @assert length(indys) == length(unique(indys))
  @assert length(unique(x[indxs])) == length(indxs)
  @assert length(unique(y[indys])) == length(indys)
  @assert length(unique(x[indys])) == 1
  @assert length(unique(y[indxs])) == 1
  return (indxs, indys)
end

indxs, indys = getindices()

function get2D(fname, str, inds)
  return hcat([h5read(fname, "Step#$(i * NS)/" * str)[inds] for i in 0:(NT÷NS)-1]...);
end

x2d = get2D(fname0, "x", indxs)
y2d = get2D(fname0, "y", indys)

NG = size(x2d, 1)
NF = size(x2d, 2)
@show NG, NP, NP÷NG^2, NT, NS, NF, dt

try
  fname = "Electrostatic2D3V_line_field_evaluations.h5part"
  phix = get2D(fname, "FIELD_EVALUATION_0", indxs)
  phiy = get2D(fname, "FIELD_EVALUATION_0", indys)

  for (Z, string) in ((phix, "phix"), (phiy, "phiy"), (x2d, "x2d"), (y2d, "y2d"))
    fig = Figure(; resolution=(600, 400))
    ax = Axis(fig[1, 1], xlabel=L"Space $[v_{th} / \Omega]$", ylabel=L"Time $[\tau_c]$")
    ts = (1:NF) .* (NS * dt) / (2pi / Wc)
    xs = (1:NG) .* Lx / NG / (vth / Wc)
    heatmap!(ax, xs, ts, Z)
    save(string * "_TX.png", fig)
  end

  filter = sin.((0.5:size(phix,2)-0.5) ./ size(phix,2) .* pi)'

  for (field, string) in ((phix, "phix"), (phiy, "phiy"))
      fig = Figure(; resolution=(600, 400))
      ax = Axis(fig[1, 1],
                xlabel=L"Wavenumber $[\Omega/v_{th}]$",
                ylabel=L"Frequency $[\Omega]$")
      Z = log10.(abs.(fft((field .* filter))))[1:end÷2, 1:end÷2];
      ks = ((1:NG) .* 2pi / Lx .* vth / Wc)[1:end÷2]
      ws = ((1:NF) .* 2pi / (dt * NT) ./ Wc)[1:end÷2]
      heatmap!(ax, ks, ws, Z)
      save(string * "_WK_c.png", fig)

      fig = Figure(; resolution=(600, 400))
      ax = Axis(fig[1, 1],
                xlabel=L"Wavenumber $[\Pi/v_{th}]$",
                ylabel=L"Frequency $[\Pi]$")
      Z = log10.(abs.(fft((field .* filter))))[1:end÷2, 1:end÷2];
      ks = ((1:NG) .* 2pi / Lx .* vth / Wp)[1:end÷2]
      ws = ((1:NF) .* 2pi / (dt * NT) ./ Wp)[1:end÷2]
      heatmap!(ax, ks, ws, Z)
      save(string * "_WK_p.png", fig)
  end
catch err
  @warn "Failed to plot electric potential line outs"
  throw(err)
end

try
  fname = "Electrostatic2D3V_line_field_deriv_evaluations.h5part"
  E0x = get2D(fname, "FIELD_DERIV_EVALUATION_0", indxs);
  E0y = get2D(fname, "FIELD_DERIV_EVALUATION_0", indys);
  E1x = get2D(fname, "FIELD_DERIV_EVALUATION_1", indxs);
  E1y = get2D(fname, "FIELD_DERIV_EVALUATION_1", indys);

  for (field, string) in ((E0x, "E0x"), (E1x, "E1x"), (E0y, "E0y"), (E1y, "E1y"))
    fig = Figure(; resolution=(600, 400))
    ax = Axis(fig[1, 1], xlabel="X", ylabel="T")
    heatmap!(ax, field)
    save(string * "_TX.png", fig)
  end

  filter = sin.((0.5:size(E0x,2)-0.5) ./ size(E0x,2) .* pi)'

  for (field, string) in ((E0x, "E0x"), (E1x, "E1x"), (E0y, "E0y"), (E1y, "E1y"))
    fig = Figure(; resolution=(600, 400))
    ax = Axis(fig[1, 1], xlabel="Wavenumber", ylabel="Frequency")
    heatmap!(ax, log10.(abs.(fft((field .* filter)))))#[1:end÷2, 1:end÷2]);
    save(string * "_WK.png", fig)
  end
catch
  @warn "Failed to plot electric field line outs"
end
end

foo()
