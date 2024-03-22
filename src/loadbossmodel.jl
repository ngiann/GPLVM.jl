function loadbossmodel()

    savedresults = JLD2.load(joinpath(artifact"bossspectralmodel","bossspectramodel.jld2"))

    @printf("Returning struct containing results trained on data provided by PPCASpectra.\n")
    @printf("The command used was \n\n\t%s\n\n", savedresults["cmd"])
    @printf("The command used at \n\n\t%s\n\n", savedresults["when"])

    return savedresults["R"]

end



function loadbossmodel_gplvm()

    savedresults = JLD2.load(joinpath(artifact"bossspectralmodel_gplvmvar","bossspectramodel_gplvar.jld2"))

    @printf("Returning struct containing results trained on data provided by PPCASpectra.\n")
    @printf("The command used was \n\n\t%s\n\n", savedresults["cmd"])

    return savedresults["R3"]

end