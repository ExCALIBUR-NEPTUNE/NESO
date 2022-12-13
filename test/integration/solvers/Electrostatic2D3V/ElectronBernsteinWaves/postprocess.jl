using HDF5, Makie, CairoMakie, FFTW, LinearAlgebra

#fname = "Electrostatic2D3V_rine_field_evaluations.h5part"
fname = "Electrostatic2D3V_line_field_deriv_evaluations.h5part"

function finddelta(z)
  uqz = unique(diff(z))
  uqz = uqz[abs.(uqz) .> 0]
  dz = uqz[findmax([sum(diff(z) .≈ i) for i in uqz])[2]]
  return dz
end

function getindices(z)
  dz = finddelta(z)
  L = length(z)
  ind = [i == 1 ? (z[i+1] - z[i] ≈ dz) : i == L ? (z[i] - z[i-1] ≈ dz) : (z[i] - z[i-1] ≈ dz) || (z[i + 1] - z[i] ≈ dz) for i in 1:L]
  return ind
end
x = h5read(fname, "Step#0/x")
y = h5read(fname, "Step#0/y")
@assert length(x) == length(y)

indxs = sort(findall(getindices(x)), by=i->x[i])
indys = sort(findall(getindices(y)), by=i->y[i])
@assert length(indxs) == length(x)÷2
@assert length(indys) == length(x)÷2

NT = 32768
NS = 4

x2d = hcat([h5read(fname, "Step#$i/x")[indxs] for i in (0, NS:NS:(NT-1)...)]...);
y2d = hcat([h5read(fname, "Step#$i/y")[indys] for i in (0, NS:NS:(NT-1)...)]...);
if true
  phix = hcat([h5read(fname, "Step#$i/FIELD_EVALUATION_0")[indxs] for i in (0, NS:NS:(NT-1)...)]...);
  phiy = hcat([h5read(fname, "Step#$i/FIELD_EVALUATION_0")[indys] for i in (0, NS:NS:(NT-1)...)]...);

  for (field, string) in ((phix, "phix"), (phiy, "phiy"), (x2d, "x2d"), (y2d, "y2d"))
    fig = Figure(; resolution=(600, 400))
    ax = Axis(fig[1, 1], xlabel="X", ylabel="T")
    heatmap!(ax, field)
    save(string * "_TX.png", fig)
  end

  filter = sin.((0.5:size(phix,2)-0.5) ./ size(phix,2) .* pi)'

  for (field, string) in ((phix, "phix"), (phiy, "phiy"))
    fig = Figure(; resolution=(600, 400))
    ax = Axis(fig[1, 1], xlabel="Wavenumber", ylabel="Frequency")
    heatmap!(ax, log10.(abs.(fft((field .* filter)')))[1:end÷2, 2:end÷2]);
    save(string * "_WK.png", fig)
  end
else
  E0x = hcat([h5read(fname, "Step#$i/FIELD_DERIV_EVALUATION_0")[indxs] for i in (0, NS:NS:(NT-1)...)]...);
  E0y = hcat([h5read(fname, "Step#$i/FIELD_DERIV_EVALUATION_0")[indys] for i in (0, NS:NS:(NT-1)...)]...);
  E1x = hcat([h5read(fname, "Step#$i/FIELD_DERIV_EVALUATION_1")[indxs] for i in (0, NS:NS:(NT-1)...)]...);
  E1y = hcat([h5read(fname, "Step#$i/FIELD_DERIV_EVALUATION_1")[indys] for i in (0, NS:NS:(NT-1)...)]...);

  for (field, string) in ((E0x, "E0x"), (E1x, "E1x"), (E0y, "E0y"), (E1y, "E1y"), (x2d, "x2d"), (y2d, "y2d"))
    fig = Figure(; resolution=(600, 400))
    ax = Axis(fig[1, 1], xlabel="X", ylabel="T")
    heatmap!(ax, field)
    save(string * "_TX.png", fig)
  end

  filter = sin.((0.5:size(E0x,2)-0.5) ./ size(E0x,2) .* pi)'

  for (field, string) in ((E0x, "E0x"), (E1x, "E1x"), (E0y, "E0y"), (E1y, "E1y"))
    fig = Figure(; resolution=(600, 400))
    ax = Axis(fig[1, 1], xlabel="Wavenumber", ylabel="Frequency")
    heatmap!(ax, log10.(abs.(fft((field .* filter)')))[1:end÷2, 2:end÷2]);
    save(string * "_WK.png", fig)
  end
end
