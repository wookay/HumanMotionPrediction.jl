export load

using PyCall

# https://github.com/dhoegh/Hawk.jl/blob/master/src/Hawk.jl

filename = abspath(joinpath(dirname(@__FILE__), "kinematics.py"))

@pyimport imp
(path, name) = dirname(filename), basename(filename)
(name, ext) = rsplit(name, '.')

(file, filename, data) = imp.find_module(name, [path])
conv = imp.load_module(name, file, filename, data)

function load(action="walking_0"; sample=Pkg.dir("HumanMotionPrediction", "samples", "walking_10000.h5"))
    conv[:load](action, sample)
end
