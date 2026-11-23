# PBS deployment checks (Aurora and Polaris)

The endpoint template must render the requested allocation into
`PBSProProvider`, including filesystem and placement options. Inspect current
queue constraints with `qstat -Q` and `qstat -Qf <queue>`; route queues may have
node-count minima that reject a smaller request.

Check maintenance reservations with `pbs_rstat` before submitting. An `M*`
reservation in state `RN` can prevent a block from starting even when
`qstat -Bf` reports `scheduling = True`, the queue is enabled and started, and
`qsub` accepts jobs. With `block_error_handler: false`, Parsl may log only one
`scale_out_facade`, making maintenance look like an ordinary queue wait.
