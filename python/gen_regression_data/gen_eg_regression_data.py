import glob
import h5py
import os.path
import sys

from NekPy.FieldUtils import Field, InputModule, ProcessModule

reg_data_fname = "regression_data.h5"


def read_nektar_fld(run_dir):
    config_fpaths = glob.glob(os.path.join(run_dir, "*.xml"))
    fld_fpath = glob.glob(os.path.join(run_dir, "*.fld"))
    assert len(fld_fpath) == 1, f"Found multiple .fld files in {run_dir}"

    # Init Field object
    field = Field(config_fpaths)

    # Read config
    InputModule.Create("xml", field, *config_fpaths).Run()

    # Read fld file
    InputModule.Create("fld", field, fld_fpath[0]).Run()

    # Compute equi-spaced points
    ProcessModule.Create("equispacedoutput", field).Run()

    # Return points in a dict to simplify output later
    ndims = field.graph.GetSpaceDimension()
    return {
        fld_name: field.GetPts(ndims + fld_idx)
        for fld_idx, fld_name in enumerate(field.session.GetVariables())
    }


def user_confirms(msg: str) -> bool:
    opts = ("y", "n")
    opt_str = " or ".join(opts)
    while (
        answer := input(f"{msg} (Choose {opt_str})").strip().lower()[:1]
    ) not in opts:
        print(f"Invalid input; choose {opt_str}")

    return answer == opts[0]


def data_dir(solver_name: str, eg_name: str):
    eg_reg_tests_dir = os.path.normpath(sys.path[0] + "/../../test/regression/examples")
    return os.path.join(eg_reg_tests_dir, solver_name, eg_name)


def gen_eg_regression_data(solver_name: str, eg_name: str, attrs={}):
    d = data_dir(solver_name, eg_name)
    os.makedirs(d, exist_ok=True)

    # ToDo Run solver with nsteps from attrs

    run_dir = os.path.join("/tmp/neso-tests/regression/examples", solver_name, eg_name)
    fld_data = read_nektar_fld(run_dir)

    pth = os.path.join(d, reg_data_fname)
    if os.path.exists(pth):
        if not user_confirms(f"Overwrite file at {pth}?"):
            print("Aborted.")
            return
    with h5py.File(pth, "w") as fh:
        for fld_name, fld_vals in fld_data.items():
            fh.create_dataset(fld_name, data=fld_vals)
        fh.attrs.update(attrs)


def read_regression_data(solver_name: str, eg_name: str):
    pth = os.path.join(data_dir(solver_name, eg_name), reg_data_fname)
    with h5py.File(pth, "r") as data:
        print(f"nsteps = {data.attrs['nsteps']}")
        for fld_name, fld_data in data.items():
            print(f"{fld_name} data: {fld_data}")


if __name__ == "__main__":
    solver_name = "SimpleSOL"
    eg_name = "1D"
    gen_eg_regression_data("SimpleSOL", "1D", attrs={"nsteps": 5000})
    read_regression_data(solver_name, eg_name)
