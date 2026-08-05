# Local fork changes

The working baseline is egg 0.11.0. rise-distance uses this fork to observe
fixed-width aggregate rewrite-scheduler state at iteration boundaries without
exposing `BackoffScheduler` internals.

Local changes are confined to `src/run.rs`, which adds `SchedulerStats`,
`RewriteScheduler::stats`, `Runner::scheduler_stats`, snapshot timing, and
focused tests for the API and existing backoff behavior. This note is the only
additional file.

These patches must be reviewed whenever egg is updated or rebased.
