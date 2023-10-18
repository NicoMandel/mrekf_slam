# TODO

1. Overwrite the functions for Hx, Hw and W_est in [Ekf_base](./src/mrekf/ekf_base.py) to correspond to the _new functions in file [Mr_ekf](./src/mrekf/mr_ekf.py)
2. clear logging / histories
3. evaluation on hitories
4. check sensor properties

## More specific Todos - 17.10.2023
1. [x] overwrite Hx for dynamic EKF - has to distribute to the right states, otherwise problematic
2. [x] make sure that dynamic lms are stored with an ID -> store it correctly in simulation log and in  the observations from the sensor
3. [x] in the simulation settings, store which lm_ids are considered dynamic for which sensor
   1. [x] Deal with the Json Infinity Values
   2. [ ] Read in the JSON again -> to use in combination with the histories
<!-- 4. [ ] Turn logs into dictionaries -> with t as the key.  -->
5. [ ] evaluate on the histories - do a run and plot on the histories and then see if we get the results we want.
   1. [ ] normal plots
   2. [ ] see if we can update the v and omega states
   3. [ ] look at what P does over time