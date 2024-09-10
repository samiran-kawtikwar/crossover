function print_optimality(crossover::Crossover)
    @printf("%30s %+15.8e\n", "Primal objective:", crossover.info.primal_objective)
    @printf("%30s %+15.8e\n", "Dual objective:", crossover.info.dual_objective)
    @printf("%30s %8.2e / %8.2e\n", "Primal feasibility (abs/rel):", crossover.info.primal_infeasibility_abs, crossover.info.primal_infeasibility_rel)
    @printf("%30s %8.2e / %8.2e\n", "Dual feasibility (abs/rel):", crossover.info.dual_infeasibility_abs, crossover.info.dual_infeasibility_rel)
    @printf("%30s %8.2e / %8.2e\n", "Gap (abs/rel):", crossover.info.duality_gap_abs, crossover.info.duality_gap_rel)
end

function print_time(timer::CrossoverTimer)
    @printf("%30s %.4f\n", "Crossover time (secs):", timer.crossover_time)
end

function print_status(crossover::Crossover)
    @printf("%30s %s\n", "Crossover status:", crossover.info.crossover_status_str)
end

function print_lp_info(lp::LinearProgramming)
    @printf("%30s %d\n", "Number of rows:", lp.nRows)
    @printf("%30s %d\n", "Number of columns:", lp.nCols)
end

function print_author()
    println("Haihao Lu, Tianhao Liu. Copyright (C) 2024.")
    # println("University of Chicago, Booth School of Business")
    println("Massachusetts Institute of Technology, Sloan School of Management")
    println("Shanghai Jiao Tong University, Antai College of Economics and Management")
end

function print_line(length::Int=80)
    println("-"^length)
end

function print_crossover_header()
    #     println(raw"""              _____ _____   ____   _____ _____  ______      ________ _____
    #              / ____|  __ \ / __ \ / ____/ ____|/ __ \ \    / /  ____|  __ \
    #    ___ _   _| |    | |__) | |  | | (___| (___ | |  | \ \  / /| |__  | |__) |
    #   / __| | | | |    |  _  /| |  | |\___ \\___ \| |  | |\ \/ / |  __| |  _  /
    #  | (__| |_| | |____| | \ \| |__| |____) |___) | |__| | \  /  | |____| | \ \
    #   \___|\__,_|\_____|_|  \_\\____/|_____/_____/ \____/   \/   |______|_|  \_\
    # """)

    println(
        raw"""   ____   ____     ___    ____    ____     ___   __     __  _____   ____  
      / ___| |  _ \   / _ \  / ___|  / ___|   / _ \  \ \   / / | ____| |  _ \ 
     | |     | |_) | | | | | \___ \  \___ \  | | | |  \ \ / /  |  _|   | |_) |
     | |___  |  _ <  | |_| |  ___) |  ___) | | |_| |   \ V /   | |___  |  _ < 
      \____| |_| \_\  \___/  |____/  |____/   \___/     \_/    |_____| |_| \_\
    """
    )

end