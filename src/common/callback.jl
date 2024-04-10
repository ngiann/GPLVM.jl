function callback(state, l)
   
    # callback function to observe training

    @printf("Iter %d, fitness = %4.6f\n", state.iter, l)
    
    return false

end