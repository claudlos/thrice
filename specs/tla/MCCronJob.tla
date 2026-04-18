------------------------------- MODULE MCCronJob -------------------------------
(*
  Model instance for TLC model-checking of CronJob.tla.

  Keeps MaxRepeat / MaxRetries small so the full state space stays
  enumerable in a few seconds on a laptop.  Adjust in MCCronJob.cfg if
  you want a broader sweep.
*)
EXTENDS CronJob

================================================================================
