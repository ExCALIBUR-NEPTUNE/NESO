using HDF5, Makie, CairoMakie, FFTW

fname = "Electrostatic2D3V_line_field_deriv_evaluations.h5part"

E1 = hcat([h5read(fname, "Step#$i/FIELD_DERIV_EVALUATION_1") for i in (0, 16:16:4095...)]...);
E0 = hcat([h5read(fname, "Step#$i/FIELD_DERIV_EVALUATION_0") for i in (0, 16:16:4095...)]...);

for (field, string) in ((E0, "E0"), (E1, "E1"))
  fig = Figure(; resolution=(600, 400))
  ax = Axis(fig[1, 1], xlabel="X", ylabel="T")
  heatmap!(ax, field)
  save(string * "_TX.png", fig)
end

filter = sin.((0.5:size(E0,2)-0.5) ./ size(E0,2) .* pi)'

for (field, string) in ((E0, "E0"), (E1, "E1"))
  fig = Figure(; resolution=(600, 400))
  ax = Axis(fig[1, 1], xlabel="Wavenumber", ylabel="Frequency")
  heatmap!(ax, log10.(abs.(fft((field .* filter)')))[1:end÷2, 2:end÷2]);
  save(string * "_WK.png", fig)
end

