------------------------------ MODULE MCBisector ------------------------------
(*
  TLC model instance for Bisector.tla.  Bounds are small so the run
  finishes in seconds on CI.  Raise MaxCommits to 8 and MaxSteps to 6
  for a heavier sweep.
*)
EXTENDS Bisector

================================================================================
