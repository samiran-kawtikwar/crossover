Base.@kwdef mutable struct CrossoverTimer
    start::Float64 = 0.0
    crossover_time::Float64 = 0.0
end

function tic!(timer::CrossoverTimer)
    timer.start = time()
end

function toc!(timer::CrossoverTimer)
    timer.crossover_time = time() - timer.start

    return timer.crossover_time
end

function get_current_runtime(timer::CrossoverTimer)
    return time() - timer.start
end

Base.getproperty(timer::CrossoverTimer, name::Symbol) =
    name == :tic! ? () -> tic!(timer) :
    name == :toc! ? () -> toc!(timer) :
    name == :get_current_runtime ? () -> get_current_runtime(timer) :
    getfield(timer, name)