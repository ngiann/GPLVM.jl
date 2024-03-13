function loadbossmodel()

    savedresults = JLD2.load(joinpath(artifact"bossspectralmodel","bossspectralmodel.jld2"))

    @printf("Returning struct containing results trained on data provided by PPCASpectra.\n")
    @printf("The command used was \n\n\t%s\n\n", savedresults["cmd"])

    return savedresults["R"]

end