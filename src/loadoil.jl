"""
    Y, L = loadoil()

Load the oil data set. There are 1000 data items, each with 12 features. 
Returns the 12×1000 data matrix `Y` and the 1000 corresponding labels `T`.
"""
function loadoil()

    data = JLD2.load(dirname(pathof(GPLVM)) * "/oil.jld2")

    Matrix(data["T"]'), data["labels"]

end