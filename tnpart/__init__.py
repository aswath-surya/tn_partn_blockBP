"""tnpart: graph-aware placement for distributed truncated-bond 2D TN simulation.

Pipeline: circuits -> peps_sim (trace) -> tngraph (snapshot) -> partition (placers)
          -> vsim (replay against costmodel) -> sweep/plots (Pareto frontiers).
See ../README.md for the chosen direction and ../instructions.md for the brief.
"""
