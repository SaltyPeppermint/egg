use core::cell::{Cell, RefCell};
use core::fmt::{self, Debug, Formatter};

use crate::no_std_prelude::*;
use log::*;

use crate::*;

/** Faciliates running rewrites over an [`EGraph`].

One use for [`EGraph`]s is as the basis of a rewriting system.
Since an egraph never "forgets" state when applying a [`Rewrite`], you
can apply many rewrites many times quite efficiently.
After the egraph is "full" (the rewrites can no longer find new
equalities) or some other condition, the egraph compactly represents
many, many equivalent expressions.
At this point, the egraph is ready for extraction (see [`Extractor`])
which can pick the represented expression that's best according to
some cost function.

This technique is called
[equality saturation](https://www.cs.cornell.edu/~ross/publications/eqsat/)
in general.
However, there can be many challenges in implementing this "outer
loop" of applying rewrites, mostly revolving around which rules to run
and when to stop.

[`Runner`] is `egg`'s provided equality saturation engine that has
reasonable defaults and implements many useful things like saturation
checking, egraph size limits, and customizable rule
[scheduling](RewriteScheduler).
Consider using [`Runner`] before rolling your own outer loop.

Here are some of the things [`Runner`] does for you:

- Saturation checking

  [`Runner`] checks to see if any of the rules added anything
  new to the [`EGraph`]. If none did, then it stops, returning
  [`StopReason::Saturated`].

- Iteration limits

  You can set a upper limit of iterations to do in case the search
  doesn't stop for some other reason. If this limit is hit, it stops with
  [`StopReason::IterationLimit`].

- [`EGraph`] size limit

  You can set a upper limit on the number of enodes in the egraph.
  If this limit is hit, it stops with
  [`StopReason::NodeLimit`].

- Time limit

  You can set a time limit on the runner.
  If this limit is hit, it stops with
  [`StopReason::TimeLimit`].

- Rule scheduling

  Some rules enable themselves, blowing up the [`EGraph`] and
  preventing other rewrites from running as many times.
  To prevent this, you can provide your own [`RewriteScheduler`] to
  govern when to run which rules.

  [`BackoffScheduler`] is the default scheduler.

[`Runner`] generates [`Iteration`]s that record some data about
each iteration.
You can add your own data to this by implementing the
[`IterationData`] trait.
[`Runner`] is generic over the [`IterationData`] that it will be in the
[`Iteration`]s, but by default it uses `()`.


# Example

```
use egg::{*, rewrite as rw};

define_language! {
    enum SimpleLanguage {
        Num(i32),
        "+" = Add([Id; 2]),
        "*" = Mul([Id; 2]),
        Symbol(Symbol),
    }
}

let rules: &[Rewrite<SimpleLanguage, ()>] = &[
    rw!("commute-add"; "(+ ?a ?b)" => "(+ ?b ?a)"),
    rw!("commute-mul"; "(* ?a ?b)" => "(* ?b ?a)"),

    rw!("add-0"; "(+ ?a 0)" => "?a"),
    rw!("mul-0"; "(* ?a 0)" => "0"),
    rw!("mul-1"; "(* ?a 1)" => "?a"),
];

pub struct MyIterData {
    smallest_so_far: usize,
}

type MyRunner = Runner<SimpleLanguage, (), MyIterData>;

impl IterationData<SimpleLanguage, ()> for MyIterData {
    fn make(runner: &MyRunner) -> Self {
        let root = runner.roots[0];
        let mut extractor = Extractor::new(&runner.egraph, AstSize);
        MyIterData {
            smallest_so_far: extractor.find_best(root).0,
        }
    }
}

let start = "(+ 0 (* 1 foo))".parse().unwrap();
// Runner is customizable in the builder pattern style.
let runner = MyRunner::new(Default::default())
    .with_iter_limit(10)
    .with_node_limit(10_000)
    .with_expr(&start)
    .with_scheduler(SimpleScheduler)
    .run(rules);

// Now we can check our iteration data to make sure that the cost only
// got better over time
for its in runner.iterations.windows(2) {
    assert!(its[0].data.smallest_so_far >= its[1].data.smallest_so_far);
}

println!(
    "Stopped after {} iterations, reason: {:?}",
    runner.iterations.len(),
    runner.stop_reason
);

```
*/
pub struct Runner<L: Language, N: Analysis<L>, IterData = ()> {
    /// The [`EGraph`] used.
    pub egraph: EGraph<L, N>,
    /// Data accumulated over each [`Iteration`].
    pub iterations: Vec<Iteration<IterData>>,
    /// The roots of expressions added by the
    /// [`with_expr`](Runner::with_expr()) method, in insertion order.
    pub roots: Vec<Id>,
    /// Why the `Runner` stopped. This will be `None` if it hasn't
    /// stopped yet.
    pub stop_reason: Option<StopReason>,
    /// A read-only snapshot of the exact scheduler state captured before the
    /// current iteration's hooks and rewrite search.
    pub scheduler_snapshot: SchedulerSnapshot,

    /// The hooks added by the
    /// [`with_hook`](Runner::with_hook()) method, in insertion order.
    #[allow(clippy::type_complexity)]
    pub hooks: Vec<Box<dyn FnMut(&mut Self) -> Result<(), String>>>,

    limits: RunnerLimits,
    scheduler: Box<dyn RewriteScheduler<L, N>>,
}

/// A function that samples the process's current live heap in bytes.
pub type MemorySampler = fn() -> u64;

/// The final memory accounting for a completed [`Runner`] run.
///
/// Every figure is an absolute process live-heap byte count, the same
/// coordinate system the configured ceiling is expressed in, so readings are
/// directly comparable across runs and processes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde-1", derive(serde::Serialize))]
pub struct MemoryReport {
    /// A fresh sample taken at the end of [`Runner::run`].
    pub final_reading: u64,
    /// Largest process live-heap reading sampled during the run.
    pub peak_reading: u64,
    /// The configured absolute process live-heap ceiling.
    pub absolute_limit: Option<u64>,
}

/// A limit-check boundary at which process memory was sampled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde-1", derive(serde::Serialize))]
#[cfg_attr(feature = "serde-1", serde(rename_all = "snake_case"))]
pub enum MemorySamplePhase {
    /// The iteration-start sample, immediately before hooks.
    BeforeHooks,
    /// An explicit sample requested while hooks are running.
    DuringHook,
    /// The sample immediately after one rule's search.
    AfterRuleSearch,
    /// The sample immediately after one rule's application.
    AfterRuleApplication,
    /// The sample after rebuild and iteration finalization.
    AfterRebuildFinalization,
}

/// Iteration-local peak process-memory telemetry.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde-1", derive(serde::Serialize))]
pub struct IterationMemoryPeak {
    /// Allocation at the decision boundary immediately before hooks.
    pub iteration_start_allocated: u64,
    /// Largest reading observed at any sampled boundary in this iteration.
    pub iteration_peak_allocated: u64,
    /// Boundary where `iteration_peak_allocated` was first observed.
    pub peak_phase: MemorySamplePhase,
    /// Rule responsible for the peak when it followed a rule operation.
    pub peak_rule: Option<Symbol>,
}

struct MemoryTracker {
    sampler: MemorySampler,
    absolute_limit: Option<u64>,
    latest: Cell<u64>,
    peak_reading: Cell<u64>,
    final_reading: Cell<Option<u64>>,
    iteration_peak: RefCell<Option<IterationMemoryPeak>>,
}

impl Debug for MemoryTracker {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("MemoryTracker")
            .field("absolute_limit", &self.absolute_limit)
            .field("latest", &self.latest)
            .field("peak_reading", &self.peak_reading)
            .field("final_reading", &self.final_reading)
            .field("iteration_peak", &self.iteration_peak)
            .finish_non_exhaustive()
    }
}

impl MemoryTracker {
    fn new(sampler: MemorySampler, absolute_limit: Option<u64>) -> Self {
        // Seed `latest` so a reading is available before the first sample.
        let initial = sampler();
        Self {
            sampler,
            absolute_limit,
            latest: Cell::new(initial),
            peak_reading: Cell::new(initial),
            final_reading: Cell::new(None),
            iteration_peak: RefCell::new(None),
        }
    }

    fn raw_sample(&self) -> u64 {
        let reading = (self.sampler)();
        self.latest.set(reading);
        self.peak_reading.set(self.peak_reading.get().max(reading));
        reading
    }

    fn begin_iteration(&self) -> u64 {
        let reading = self.raw_sample();
        self.iteration_peak.replace(Some(IterationMemoryPeak {
            iteration_start_allocated: reading,
            iteration_peak_allocated: reading,
            peak_phase: MemorySamplePhase::BeforeHooks,
            peak_rule: None,
        }));
        reading
    }

    fn sample_at(&self, phase: MemorySamplePhase, rule: Option<Symbol>) -> u64 {
        let reading = self.raw_sample();
        let mut peak = self.iteration_peak.borrow_mut();
        if let Some(peak) = peak.as_mut()
            && reading > peak.iteration_peak_allocated
        {
            peak.iteration_peak_allocated = reading;
            peak.peak_phase = phase;
            peak.peak_rule = rule;
        }
        reading
    }

    fn iteration_peak(&self) -> Option<IterationMemoryPeak> {
        self.iteration_peak.borrow().clone()
    }

    fn finish(&self) {
        self.final_reading.set(Some(self.raw_sample()));
    }

    fn final_report(&self) -> Option<MemoryReport> {
        self.final_reading.get().map(|final_reading| MemoryReport {
            final_reading,
            peak_reading: self.peak_reading.get(),
            absolute_limit: self.absolute_limit,
        })
    }
}

/// Describes the limits that would stop a [`Runner`].
#[derive(Debug)]
pub struct RunnerLimits {
    iter_limit: usize,
    node_limit: usize,
    time_limit: Duration,
    start_time: Option<Instant>,
    memory: Option<MemoryTracker>,
}

impl RunnerLimits {
    fn check_reading<L, N>(
        &self,
        iteration: usize,
        egraph: &EGraph<L, N>,
        memory_reading: Option<u64>,
    ) -> RunnerResult<()>
    where
        L: Language,
        N: Analysis<L>,
    {
        let elapsed = self.start_time.unwrap().elapsed();
        if elapsed > self.time_limit {
            return Err(StopReason::TimeLimit(elapsed.as_secs_f64()));
        }

        let size = egraph.total_size();
        if size > self.node_limit {
            return Err(StopReason::NodeLimit(size));
        }

        if iteration >= self.iter_limit {
            return Err(StopReason::IterationLimit(iteration));
        }

        if let (Some(memory), Some(reading)) = (&self.memory, memory_reading)
            && memory.absolute_limit.is_some_and(|limit| reading >= limit)
        {
            return Err(StopReason::MemoryLimit(reading));
        }

        Ok(())
    }

    fn begin_iteration<L, N>(&self, iteration: usize, egraph: &EGraph<L, N>) -> RunnerResult<()>
    where
        L: Language,
        N: Analysis<L>,
    {
        let reading = self.memory.as_ref().map(MemoryTracker::begin_iteration);
        self.check_reading(iteration, egraph, reading)
    }

    /// Check limits using one process-memory reading attributed to `phase`.
    pub fn check_limits_at<L, N>(
        &self,
        iteration: usize,
        egraph: &EGraph<L, N>,
        phase: MemorySamplePhase,
        rule: Option<Symbol>,
    ) -> RunnerResult<()>
    where
        L: Language,
        N: Analysis<L>,
    {
        let reading = self
            .memory
            .as_ref()
            .map(|memory| memory.sample_at(phase, rule));
        self.check_reading(iteration, egraph, reading)
    }

    /// Compatibility limit check for custom schedulers that cannot attribute
    /// a boundary to a specific rule.
    pub fn check_limits<L, N>(&self, iteration: usize, egraph: &EGraph<L, N>) -> RunnerResult<()>
    where
        L: Language,
        N: Analysis<L>,
    {
        self.check_limits_at(iteration, egraph, MemorySamplePhase::AfterRuleSearch, None)
    }
}

impl<L, N> Default for Runner<L, N, ()>
where
    L: Language,
    N: Analysis<L> + Default,
{
    fn default() -> Self {
        Runner::new(N::default())
    }
}

impl<L, N, IterData> Debug for Runner<L, N, IterData>
where
    L: Language,
    N: Analysis<L>,
    IterData: Debug,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        // Use an exhaustive pattern match to ensure the Debug implementation and the struct stay in sync.
        let Runner {
            egraph,
            iterations,
            roots,
            stop_reason,
            scheduler_snapshot,
            hooks,
            limits,
            scheduler: _,
        } = self;

        f.debug_struct("Runner")
            .field("egraph", egraph)
            .field("iterations", iterations)
            .field("roots", roots)
            .field("stop_reason", stop_reason)
            .field("scheduler_snapshot", scheduler_snapshot)
            .field("hooks", &vec![format_args!("<dyn FnMut ..>"); hooks.len()])
            .field("limits", limits)
            .field("scheduler", &format_args!("<dyn RewriteScheduler ..>"))
            .finish()
    }
}

/// Error returned by [`Runner`] when it stops.
///
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde-1", derive(serde::Serialize))]
pub enum StopReason {
    /// The egraph saturated, i.e., there was an iteration where we
    /// didn't learn anything new from applying the rules.
    Saturated,
    /// The iteration limit was hit. The data is the iteration limit.
    IterationLimit(usize),
    /// The enode limit was hit. The data is the enode limit.
    NodeLimit(usize),
    /// The time limit was hit. The data is the time limit in seconds.
    TimeLimit(f64),
    /// The absolute process live-heap limit was reached or exceeded. The data is the
    /// observed absolute live heap in bytes.
    MemoryLimit(u64),
    /// Some other reason to stop.
    Other(String),
}

/// A report containing data about an entire [`Runner`] run.
///
/// This is basically a summary of the [`Iteration`] data,
/// but summed across iterations.
/// See [`Iteration`] docs for details about fields.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde-1", derive(serde::Serialize))]
#[non_exhaustive]
#[allow(missing_docs)]
pub struct Report {
    /// The number of iterations this runner performed.
    pub iterations: usize,
    pub stop_reason: StopReason,
    pub egraph_nodes: usize,
    pub egraph_classes: usize,
    pub memo_size: usize,
    pub rebuilds: usize,
    pub total_time: f64,
    pub search_time: f64,
    pub apply_time: f64,
    pub rebuild_time: f64,
}

impl core::fmt::Display for Report {
    #[rustfmt::skip]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(f, "Runner report")?;
        writeln!(f, "=============")?;
        writeln!(f, "  Stop reason: {:?}", self.stop_reason)?;
        writeln!(f, "  Iterations: {}", self.iterations)?;
        writeln!(f, "  Egraph size: {} nodes, {} classes, {} memo", self.egraph_nodes, self.egraph_classes, self.memo_size)?;
        writeln!(f, "  Rebuilds: {}", self.rebuilds)?;
        writeln!(f, "  Total time: {}", self.total_time)?;
        let pct = |part: f64| if self.total_time > 0.0 { part / self.total_time } else { 0.0 };
        writeln!(f, "    Search:  ({:.2}) {}", pct(self.search_time), self.search_time)?;
        writeln!(f, "    Apply:   ({:.2}) {}", pct(self.apply_time), self.apply_time)?;
        writeln!(f, "    Rebuild: ({:.2}) {}", pct(self.rebuild_time), self.rebuild_time)?;
        Ok(())
    }
}

/// Data generated by running a [`Runner`] one iteration.
///
/// If the `serde-1` feature is enabled, this implements
/// [`serde::Serialize`][ser], which is useful if you want to output
/// this as a JSON or some other format.
///
/// [ser]: https://docs.rs/serde/latest/serde/trait.Serialize.html
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde-1", derive(serde::Serialize))]
#[non_exhaustive]
pub struct Iteration<IterData> {
    /// The number of enodes in the egraph at the start of this
    /// iteration.
    pub egraph_nodes: usize,
    /// The number of eclasses in the egraph at the start of this
    /// iteration.
    pub egraph_classes: usize,
    /// A map from rule name to number of times it was _newly_ applied
    /// in this iteration.
    pub applied: IndexMap<Symbol, usize>,
    /// Seconds spent running hooks.
    pub hook_time: f64,
    /// Seconds spent searching in this iteration.
    pub search_time: f64,
    /// Seconds spent applying rules in this iteration.
    pub apply_time: f64,
    /// Seconds spent [`rebuild`](EGraph::rebuild())ing
    /// the egraph in this iteration.
    pub rebuild_time: f64,
    /// Total time spent in this iteration, including data generation time.
    pub total_time: f64,
    /// The user provided annotation for this iteration
    pub data: IterData,
    /// The number of rebuild iterations done after this iteration completed.
    pub n_rebuilds: usize,
    /// If the runner stopped on this iterations, this is the reason
    pub stop_reason: Option<StopReason>,
}

/// Type alias for the result of a [`Runner`].
pub type RunnerResult<T> = core::result::Result<T, StopReason>;

impl<L, N, IterData> Runner<L, N, IterData>
where
    L: Language,
    N: Analysis<L>,
    IterData: IterationData<L, N>,
{
    /// Create a new `Runner` with the given analysis and default parameters.
    pub fn new(analysis: N) -> Self {
        Self::new_internal(analysis, None)
    }

    /// Create a new `Runner` whose process memory is sampled and optionally
    /// limited. Readings and the limit are absolute process live-heap bytes.
    pub fn new_with_memory_tracker(
        analysis: N,
        sampler: MemorySampler,
        absolute_limit: Option<u64>,
    ) -> Self {
        Self::new_internal(analysis, Some(MemoryTracker::new(sampler, absolute_limit)))
    }

    fn new_internal(analysis: N, memory: Option<MemoryTracker>) -> Self {
        Self {
            limits: RunnerLimits {
                iter_limit: 30,
                node_limit: 10_000,
                time_limit: Duration::from_secs(5),
                start_time: None,
                memory,
            },
            egraph: EGraph::new(analysis),
            roots: vec![],
            iterations: vec![],
            stop_reason: None,
            scheduler_snapshot: SchedulerSnapshot::default(),
            hooks: vec![],
            scheduler: Box::new(BackoffScheduler::default()),
        }
    }

    /// Return the most recent absolute sample already taken by this runner.
    #[must_use]
    pub fn memory_reading(&self) -> Option<u64> {
        self.limits
            .memory
            .as_ref()
            .map(|memory| memory.latest.get())
    }

    /// Take, store, and return a fresh absolute process-memory sample.
    pub fn sample_memory(&self) -> Option<u64> {
        self.limits
            .memory
            .as_ref()
            .map(|memory| memory.sample_at(MemorySamplePhase::DuringHook, None))
    }

    /// Return the current iteration's peak-memory telemetry.
    #[must_use]
    pub fn iteration_memory_peak(&self) -> Option<IterationMemoryPeak> {
        self.limits
            .memory
            .as_ref()
            .and_then(MemoryTracker::iteration_peak)
    }

    /// Return the configured absolute process live-heap ceiling.
    #[must_use]
    pub fn absolute_memory_limit(&self) -> Option<u64> {
        self.limits
            .memory
            .as_ref()
            .and_then(|memory| memory.absolute_limit)
    }

    /// Return the final report after [`Runner::run`] has finalized memory.
    #[must_use]
    pub fn final_memory_report(&self) -> Option<MemoryReport> {
        self.limits
            .memory
            .as_ref()
            .and_then(MemoryTracker::final_report)
    }

    /// Sets the iteration limit. Default: 30
    pub fn with_iter_limit(mut self, iter_limit: usize) -> Self {
        self.limits.iter_limit = iter_limit;
        self
    }

    /// Sets the egraph size limit (in enodes). Default: 10,000
    pub fn with_node_limit(mut self, node_limit: usize) -> Self {
        self.limits.node_limit = node_limit;
        self
    }

    /// Sets the runner time limit. Default: 5 seconds
    pub fn with_time_limit(mut self, time_limit: Duration) -> Self {
        self.limits.time_limit = time_limit;
        self
    }

    /// Add a hook to instrument or modify the behavior of a [`Runner`].
    /// Each hook will run at the beginning of each iteration, i.e. before
    /// all the rewrites.
    ///
    /// If your hook modifies the e-graph, make sure to call
    /// [`rebuild`](EGraph::rebuild()).
    ///
    /// # Example
    /// ```
    /// # use egg::*;
    /// let rules: &[Rewrite<SymbolLang, ()>] = &[
    ///     rewrite!("commute-add"; "(+ ?a ?b)" => "(+ ?b ?a)"),
    ///     // probably some others ...
    /// ];
    ///
    /// Runner::<SymbolLang, ()>::default()
    ///     .with_expr(&"(+ 5 2)".parse().unwrap())
    ///     .with_hook(|runner| {
    ///          println!("Egraph is this big: {}", runner.egraph.total_size());
    ///          Ok(())
    ///     })
    ///     .run(rules);
    /// ```
    pub fn with_hook<F>(mut self, hook: F) -> Self
    where
        F: FnMut(&mut Self) -> Result<(), String> + 'static,
    {
        self.hooks.push(Box::new(hook));
        self
    }

    /// Change out the [`RewriteScheduler`] used by this [`Runner`].
    /// The default one is [`BackoffScheduler`].
    ///
    pub fn with_scheduler(self, scheduler: impl RewriteScheduler<L, N> + 'static) -> Self {
        let scheduler = Box::new(scheduler);
        Self { scheduler, ..self }
    }

    /// Add an expression to the egraph to be run.
    ///
    /// The eclass id of this addition will be recorded in the
    /// [`roots`](Runner::roots) field, ordered by
    /// insertion order.
    pub fn with_expr(mut self, expr: &RecExpr<L>) -> Self {
        let id = self.egraph.add_expr(expr);
        self.roots.push(id);
        self
    }

    /// Replace the [`EGraph`] of this `Runner`.
    pub fn with_egraph(self, egraph: EGraph<L, N>) -> Self {
        Self { egraph, ..self }
    }

    /// Run this `Runner` until it stops.
    /// After this, the field
    /// [`stop_reason`](Runner::stop_reason) is guaranteed to be
    /// set.
    pub fn run<'a, R>(mut self, rules: R) -> Self
    where
        R: IntoIterator<Item = &'a Rewrite<L, N>>,
        L: 'a,
        N: 'a,
    {
        let rules: Vec<&Rewrite<L, N>> = rules.into_iter().collect();
        check_rules(&rules);
        self.egraph.rebuild();
        loop {
            let iter = self.run_one(&rules);
            self.iterations.push(iter);
            let stop_reason = self.iterations.last().unwrap().stop_reason.clone();
            // we need to check_limits after the iteration is complete to check for iter_limit
            if let Some(stop_reason) = stop_reason.or_else(|| self.check_limits().err()) {
                info!("Stopping: {:?}", stop_reason);
                self.stop_reason = Some(stop_reason);
                break;
            }
        }

        assert!(!self.iterations.is_empty());
        assert!(self.stop_reason.is_some());
        if let Some(memory) = &self.limits.memory {
            memory.finish();
        }
        self
    }

    /// Enable explanations for this runner's egraph.
    /// This allows the runner to explain why two expressions are
    /// equivalent with the [`explain_equivalence`](Runner::explain_equivalence) function.
    pub fn with_explanations_enabled(mut self) -> Self {
        self.egraph = self.egraph.with_explanations_enabled();
        self
    }

    /// By default, egg runs a greedy algorithm to reduce the size of resulting explanations (without complexity overhead).
    /// Use this function to turn this algorithm off.
    pub fn without_explanation_length_optimization(mut self) -> Self {
        self.egraph = self.egraph.without_explanation_length_optimization();
        self
    }

    /// By default, egg runs a greedy algorithm to reduce the size of resulting explanations (without complexity overhead).
    /// Use this function to turn this algorithm on again if you have turned it off.
    pub fn with_explanation_length_optimization(mut self) -> Self {
        self.egraph = self.egraph.with_explanation_length_optimization();
        self
    }

    /// Disable explanations for this runner's egraph.
    pub fn with_explanations_disabled(mut self) -> Self {
        self.egraph = self.egraph.with_explanations_disabled();
        self
    }

    /// Calls [`EGraph::explain_equivalence`](EGraph::explain_equivalence()).
    pub fn explain_equivalence(&mut self, left: &RecExpr<L>, right: &RecExpr<L>) -> Explanation<L> {
        self.egraph.explain_equivalence(left, right)
    }

    /// Get an explanation for why an expression matches a pattern.
    pub fn explain_matches(
        &mut self,
        left: &RecExpr<L>,
        right: &PatternAst<L>,
        subst: &Subst,
    ) -> Explanation<L> {
        self.egraph.explain_matches(left, right, subst)
    }

    /// Prints some information about a runners run.
    #[cfg(feature = "std")]
    pub fn print_report(&self) {
        println!("{}", self.report())
    }

    /// Creates a [`Report`] summarizing this `Runner`s run.
    pub fn report(&self) -> Report {
        Report {
            stop_reason: self.stop_reason.clone().unwrap(),
            iterations: self.iterations.len(),
            egraph_nodes: self.egraph.total_number_of_nodes(),
            egraph_classes: self.egraph.number_of_classes(),
            memo_size: self.egraph.total_size(),
            rebuilds: self.iterations.iter().map(|i| i.n_rebuilds).sum(),
            search_time: self.iterations.iter().map(|i| i.search_time).sum(),
            apply_time: self.iterations.iter().map(|i| i.apply_time).sum(),
            rebuild_time: self.iterations.iter().map(|i| i.rebuild_time).sum(),
            total_time: self.iterations.iter().map(|i| i.total_time).sum(),
        }
    }

    fn run_one(&mut self, rules: &[&Rewrite<L, N>]) -> Iteration<IterData> {
        assert!(self.stop_reason.is_none());

        let i = self.iterations.len();
        info!("\nIteration {}", i);

        self.try_start();
        let mut result = self.limits.begin_iteration(i, &self.egraph);
        self.scheduler_snapshot = self.scheduler.snapshot(i, rules);

        let egraph_nodes = self.egraph.total_size();
        let egraph_classes = self.egraph.number_of_classes();

        let hook_time = Instant::now();
        let mut hooks = core::mem::take(&mut self.hooks);
        result = result.and_then(|_| {
            hooks
                .iter_mut()
                .try_for_each(|hook| hook(self).map_err(StopReason::Other))
        });
        self.hooks = hooks;
        let hook_time = hook_time.elapsed().as_secs_f64();

        let egraph_nodes_after_hooks = self.egraph.total_size();
        let egraph_classes_after_hooks = self.egraph.number_of_classes();

        trace!("EGraph {:?}", self.egraph.dump());

        let start_time = Instant::now();

        let mut matches = Vec::new();
        let mut applied = IndexMap::default();
        result = result.and_then(|_| {
            matches = self
                .scheduler
                .search_rewrites(i, &self.egraph, rules, &self.limits)?;
            Ok(())
            // rules.iter().try_for_each(|rw| {
            //     let ms = self.scheduler.search_rewrite(i, &self.egraph, rw);
            //     matches.push(ms);
            //     self.check_limits()
            // })
        });

        let search_time = start_time.elapsed().as_secs_f64();
        info!("Search time: {}", search_time);

        let apply_time = Instant::now();

        result = result.and_then(|_| {
            rules.iter().zip(matches).try_for_each(|(rw, ms)| {
                let total_matches: usize = ms.iter().map(|m| m.substs.len()).sum();
                debug!("Applying {} {} times", rw.name, total_matches);

                let actually_matched = self.scheduler.apply_rewrite(i, &mut self.egraph, rw, ms);
                if actually_matched > 0 {
                    if let Some(count) = applied.get_mut(&rw.name) {
                        *count += actually_matched;
                    } else {
                        applied.insert(rw.name.to_owned(), actually_matched);
                    }
                    debug!("Applied {} {} times", rw.name, actually_matched);
                }
                self.check_limits_at(MemorySamplePhase::AfterRuleApplication, Some(rw.name))
            })
        });

        let apply_time = apply_time.elapsed().as_secs_f64();
        info!("Apply time: {}", apply_time);

        let rebuild_time = Instant::now();
        let n_rebuilds = self.egraph.rebuild();
        if self.egraph.are_explanations_enabled() {
            debug_assert!(self.egraph.check_each_explain(rules));
        }

        let rebuild_time = rebuild_time.elapsed().as_secs_f64();
        info!("Rebuild time: {}", rebuild_time);
        info!(
            "Size: n={}, e={}",
            self.egraph.total_size(),
            self.egraph.number_of_classes()
        );

        let finalized = self.check_limits_at(MemorySamplePhase::AfterRebuildFinalization, None);
        if result.is_ok() {
            result = finalized;
        }

        let can_be_saturated = applied.is_empty()
            && self.scheduler.can_stop(i)
            // now make sure the hooks didn't do anything
            && (egraph_nodes == egraph_nodes_after_hooks)
            && (egraph_classes == egraph_classes_after_hooks)
            // now make sure that conditional rules (which might add
            // nodes without applying) didn't do anything
            && (egraph_nodes == self.egraph.total_size())
            && (egraph_classes == self.egraph.number_of_classes());

        if can_be_saturated {
            result = result.and(Err(StopReason::Saturated))
        }

        Iteration {
            applied,
            egraph_nodes,
            egraph_classes,
            hook_time,
            search_time,
            apply_time,
            rebuild_time,
            n_rebuilds,
            data: IterData::make(self),
            total_time: start_time.elapsed().as_secs_f64(),
            stop_reason: result.err(),
        }
    }

    fn try_start(&mut self) {
        self.limits.start_time.get_or_insert_with(Instant::now);
    }

    fn check_limits(&self) -> RunnerResult<()> {
        let reading = self.memory_reading();
        self.limits
            .check_reading(self.iterations.len(), &self.egraph, reading)
    }

    fn check_limits_at(&self, phase: MemorySamplePhase, rule: Option<Symbol>) -> RunnerResult<()> {
        self.limits
            .check_limits_at(self.iterations.len(), &self.egraph, phase, rule)
    }
}

fn check_rules<L, N>(rules: &[&Rewrite<L, N>]) {
    let mut name_counts = IndexMap::default();
    for rw in rules {
        *name_counts.entry(rw.name).or_default() += 1
    }

    name_counts.retain(|_, count: &mut usize| *count > 1);
    if !name_counts.is_empty() {
        #[cfg(feature = "std")]
        eprintln!("WARNING: Duplicated rule names may affect rule reporting and scheduling.");
        log::warn!("Duplicated rule names may affect rule reporting and scheduling.");
        for (name, &count) in name_counts.iter() {
            assert!(count > 1);
            #[cfg(feature = "std")]
            eprintln!("Rule '{}' appears {} times", name, count);
            log::warn!("Rule '{}' appears {} times", name, count);
        }
    }
}

/** A way to customize how a [`Runner`] runs [`Rewrite`]s.

This gives you a way to prevent certain [`Rewrite`]s from exploding
the [`EGraph`] and dominating how much time is spent while running the
[`Runner`].

*/
#[allow(unused_variables)]
pub trait RewriteScheduler<L, N>
where
    L: Language,
    N: Analysis<L>,
{
    /// Legacy aggregate-only introspection. New code should use
    /// [`Self::snapshot`], which includes the exact per-rule search state.
    fn stats(&self, iteration: usize) -> SchedulerStats {
        let _ = iteration;
        SchedulerStats::default()
    }

    /// Return a read-only snapshot of scheduler state for the iteration that
    /// is about to search.
    ///
    /// The default identifies itself as a generic scheduler and reports every
    /// rule active with no finite match threshold. A model may deliberately
    /// support that representation, but a manifest requiring backoff state
    /// must reject it.
    fn snapshot(&self, iteration: usize, rewrites: &[&Rewrite<L, N>]) -> SchedulerSnapshot {
        SchedulerSnapshot::generic(iteration, rewrites)
    }

    /// Whether or not the [`Runner`] is allowed
    /// to say it has saturated.
    ///
    /// This is only called when the runner is otherwise saturated.
    /// Default implementation just returns `true`.
    fn can_stop(&mut self, iteration: usize) -> bool {
        true
    }

    /// A hook allowing you to customize rewrite searching behavior.
    /// Useful to implement rule management.
    ///
    /// Default implementation just calls
    /// [`Rewrite::search`](Rewrite::search()).
    fn search_rewrite<'a>(
        &mut self,
        iteration: usize,
        egraph: &EGraph<L, N>,
        rewrite: &'a Rewrite<L, N>,
    ) -> Vec<SearchMatches<'a, L>> {
        rewrite.search(egraph)
    }

    /// A hook allowing you to customize rewrite searching behavior
    /// across rewrites.
    ///
    /// Default implementation calls
    /// [`Self::search_rewrite`] for each rewrite,
    /// and checks [`RunnerLimits::check_limits`] after each.
    ///
    /// Returning an error will stop the runner.
    ///
    /// You might use this to implement parallel rule application:
    /// ```
    /// # use egg::*;
    /// pub struct ParallelRewriteScheduler;
    /// impl RewriteScheduler<SymbolLang, ()> for ParallelRewriteScheduler {
    ///     fn search_rewrites<'a>(
    ///         &mut self,
    ///         iteration: usize,
    ///         egraph: &EGraph<SymbolLang, ()>,
    ///         rewrites: &[&'a Rewrite<SymbolLang, ()>],
    ///         _limits: &RunnerLimits,
    ///     ) -> RunnerResult<Vec<Vec<SearchMatches<'a, SymbolLang>>>> {
    ///         // this implementation just ignores the limits
    ///         // fake `par_map` to enforce Send + Sync, in real life use rayon
    ///         fn par_map<T, F, T2>(slice: &[T], f: F) -> Vec<T2>
    ///         where
    ///             T: Send + Sync,
    ///             F: Fn(&T) -> T2 + Send + Sync,
    ///             T2: Send + Sync,
    ///         {
    ///             slice.iter().map(f).collect()
    ///         }
    ///         Ok(par_map(rewrites, |rw| rw.search(egraph)))
    ///     }
    /// }
    /// ```
    fn search_rewrites<'a>(
        &mut self,
        iteration: usize,
        egraph: &EGraph<L, N>,
        rewrites: &[&'a Rewrite<L, N>],
        limits: &RunnerLimits,
    ) -> RunnerResult<Vec<Vec<SearchMatches<'a, L>>>> {
        let mut matches = Vec::new();
        for rw in rewrites {
            let ms = self.search_rewrite(iteration, egraph, rw);
            matches.push(ms);
            limits.check_limits_at(
                iteration,
                egraph,
                MemorySamplePhase::AfterRuleSearch,
                Some(rw.name),
            )?;
        }
        Ok(matches)
    }

    /// A hook allowing you to customize rewrite application behavior.
    /// Useful to implement rule management.
    ///
    /// Default implementation just calls
    /// [`Rewrite::apply`](Rewrite::apply())
    /// and returns number of new applications.
    fn apply_rewrite(
        &mut self,
        iteration: usize,
        egraph: &mut EGraph<L, N>,
        rewrite: &Rewrite<L, N>,
        matches: Vec<SearchMatches<L>>,
    ) -> usize {
        rewrite.apply(egraph, &matches).len()
    }
}

/// A very simple [`RewriteScheduler`] that runs every rewrite every
/// time.
///
/// Using this is basically turning off rule scheduling.
/// It uses the default implementation for all [`RewriteScheduler`]
/// methods.
///
/// This is not the default scheduler; choose it with the
/// [`with_scheduler`](Runner::with_scheduler())
/// method.
///
#[derive(Debug)]
pub struct SimpleScheduler;

impl<L, N> RewriteScheduler<L, N> for SimpleScheduler
where
    L: Language,
    N: Analysis<L>,
{
}

/// Legacy fixed-width aggregate scheduler state.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SchedulerStats {
    /// Number of rules whose ban expires after this iteration.
    pub n_banned: usize,
    /// Number of rules whose ban expires exactly at this iteration.
    pub n_unbanned_this_iter: usize,
    /// Fewest iterations remaining among currently banned rules.
    pub min_ban_remaining: usize,
    /// Total number of bans issued across all rules.
    pub total_times_banned: usize,
}

/// Per-rule state governing one upcoming rewrite search.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde-1", derive(serde::Serialize))]
pub struct SchedulerRuleState {
    /// Raw egg rule name.
    pub name: Symbol,
    /// Whether this rule will be searched in the upcoming iteration.
    pub will_search: bool,
    /// Whether a previous ban expires exactly at this iteration.
    pub newly_unbanned: bool,
    /// Number of times the scheduler has banned this rule.
    pub times_banned: usize,
    /// Remaining iterations in the ban, or zero when active.
    pub ban_remaining: usize,
    /// Exact effective match threshold used by `search_rewrite`.
    pub match_limit: usize,
    /// Stable logarithmic representation of `match_limit`.
    pub log2_match_limit: f64,
}

/// Aggregate and per-rule scheduler state for one upcoming iteration.
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "serde-1", derive(serde::Serialize))]
pub struct SchedulerSnapshot {
    /// Scheduler representation name used for manifest compatibility.
    pub scheduler: &'static str,
    /// Number of rules active for the upcoming search.
    pub n_active: usize,
    /// Number of rules whose ban expires after this iteration.
    pub n_banned: usize,
    /// Number of rules whose previous ban expires exactly this iteration.
    pub n_newly_unbanned: usize,
    /// Fewest iterations remaining among currently banned rules.
    pub min_ban_remaining: usize,
    /// Total number of bans issued across all rules.
    pub total_times_banned: usize,
    /// Largest effective log2 match limit among active rules.
    pub max_active_log2_match_limit: f64,
    /// `log2(1 + sum(active match limits))`.
    pub log2_active_match_limit_sum: f64,
    /// Largest ban count among active rules.
    pub max_active_times_banned: usize,
    /// State for every rule, in the rewrite slice's deterministic order.
    pub rules: Vec<SchedulerRuleState>,
}

impl SchedulerSnapshot {
    fn generic<L, N>(iteration: usize, rewrites: &[&Rewrite<L, N>]) -> Self
    where
        L: Language,
        N: Analysis<L>,
    {
        let rules = rewrites
            .iter()
            .map(|rewrite| SchedulerRuleState {
                name: rewrite.name,
                will_search: true,
                newly_unbanned: false,
                times_banned: 0,
                ban_remaining: 0,
                match_limit: usize::MAX,
                log2_match_limit: match_limit_log2(usize::MAX),
            })
            .collect();
        let mut result = Self {
            scheduler: "generic",
            rules,
            ..Self::default()
        };
        result.recompute_aggregates();
        let _ = iteration;
        result
    }

    fn recompute_aggregates(&mut self) {
        self.n_active = self.rules.iter().filter(|rule| rule.will_search).count();
        self.n_banned = self.rules.len() - self.n_active;
        self.n_newly_unbanned = self.rules.iter().filter(|rule| rule.newly_unbanned).count();
        self.min_ban_remaining = self
            .rules
            .iter()
            .filter_map(|rule| (rule.ban_remaining > 0).then_some(rule.ban_remaining))
            .min()
            .unwrap_or(0);
        self.total_times_banned = self.rules.iter().map(|rule| rule.times_banned).sum();
        self.max_active_log2_match_limit = self
            .rules
            .iter()
            .filter(|rule| rule.will_search)
            .map(|rule| rule.log2_match_limit)
            .fold(0.0, f64::max);
        let active_limit_sum = self
            .rules
            .iter()
            .filter(|rule| rule.will_search)
            .map(|rule| rule.match_limit as f64)
            .sum::<f64>();
        self.log2_active_match_limit_sum = (1.0 + active_limit_sum).log2();
        self.max_active_times_banned = self
            .rules
            .iter()
            .filter(|rule| rule.will_search)
            .map(|rule| rule.times_banned)
            .max()
            .unwrap_or(0);
    }
}

/// Saturating effective threshold shared by introspection and search.
fn effective_match_limit(base: usize, times_banned: usize) -> usize {
    if base == 0 {
        return 0;
    }
    let Ok(shift) = u32::try_from(times_banned) else {
        return usize::MAX;
    };
    if shift >= usize::BITS {
        return usize::MAX;
    }
    base.checked_mul(1_usize << shift).unwrap_or(usize::MAX)
}

fn match_limit_log2(limit: usize) -> f64 {
    if limit == 0 {
        0.0
    } else {
        (limit as f64).log2()
    }
}

/// A [`RewriteScheduler`] that implements exponentional rule backoff.
///
/// For each rewrite, there exists a configurable initial match limit.
/// If a rewrite search yield more than this limit, then we ban this
/// rule for number of iterations, double its limit, and double the time
/// it will be banned next time.
///
/// This seems effective at preventing explosive rules like
/// associativity from taking an unfair amount of resources.
///
/// [`BackoffScheduler`] is configurable in the builder-pattern style.
///
#[derive(Debug)]
pub struct BackoffScheduler {
    default_match_limit: usize,
    default_ban_length: usize,
    stats: IndexMap<Symbol, RuleStats>,
}

#[derive(Debug)]
struct RuleStats {
    times_applied: usize,
    banned_until: usize,
    times_banned: usize,
    match_limit: usize,
    ban_length: usize,
}

impl BackoffScheduler {
    /// Set the initial match limit after which a rule will be banned.
    /// Default: 1,000
    pub fn with_initial_match_limit(mut self, limit: usize) -> Self {
        self.default_match_limit = limit;
        self
    }

    /// Set the initial ban length.
    /// Default: 5 iterations
    pub fn with_ban_length(mut self, ban_length: usize) -> Self {
        self.default_ban_length = ban_length;
        self
    }

    fn rule_stats(&mut self, name: Symbol) -> &mut RuleStats {
        if self.stats.contains_key(&name) {
            &mut self.stats[&name]
        } else {
            self.stats.entry(name).or_insert(RuleStats {
                times_applied: 0,
                banned_until: 0,
                times_banned: 0,
                match_limit: self.default_match_limit,
                ban_length: self.default_ban_length,
            })
        }
    }

    /// Never ban a particular rule.
    pub fn do_not_ban(mut self, name: impl Into<Symbol>) -> Self {
        self.rule_stats(name.into()).match_limit = usize::MAX;
        self
    }

    /// Set the initial match limit for a rule.
    pub fn rule_match_limit(mut self, name: impl Into<Symbol>, limit: usize) -> Self {
        self.rule_stats(name.into()).match_limit = limit;
        self
    }

    /// Set the initial ban length for a rule.
    pub fn rule_ban_length(mut self, name: impl Into<Symbol>, length: usize) -> Self {
        self.rule_stats(name.into()).ban_length = length;
        self
    }
}

impl Default for BackoffScheduler {
    fn default() -> Self {
        Self {
            stats: Default::default(),
            default_match_limit: 1_000,
            default_ban_length: 5,
        }
    }
}

impl<L, N> RewriteScheduler<L, N> for BackoffScheduler
where
    L: Language,
    N: Analysis<L>,
{
    fn stats(&self, iteration: usize) -> SchedulerStats {
        let mut result = SchedulerStats::default();
        for stats in self.stats.values() {
            result.total_times_banned += stats.times_banned;
            if stats.banned_until > iteration {
                result.n_banned += 1;
                let remaining = stats.banned_until - iteration;
                result.min_ban_remaining = if result.min_ban_remaining == 0 {
                    remaining
                } else {
                    result.min_ban_remaining.min(remaining)
                };
            } else if stats.times_banned > 0 && stats.banned_until == iteration {
                result.n_unbanned_this_iter += 1;
            }
        }
        result
    }

    fn snapshot(&self, iteration: usize, rewrites: &[&Rewrite<L, N>]) -> SchedulerSnapshot {
        let rules = rewrites
            .iter()
            .map(|rewrite| {
                let stored = self.stats.get(&rewrite.name);
                let banned_until = stored.map_or(0, |stats| stats.banned_until);
                let times_banned = stored.map_or(0, |stats| stats.times_banned);
                let base_limit = stored.map_or(self.default_match_limit, |stats| stats.match_limit);
                let will_search = banned_until <= iteration;
                let match_limit = effective_match_limit(base_limit, times_banned);
                SchedulerRuleState {
                    name: rewrite.name,
                    will_search,
                    newly_unbanned: times_banned > 0 && banned_until == iteration,
                    times_banned,
                    ban_remaining: banned_until.saturating_sub(iteration),
                    match_limit,
                    log2_match_limit: match_limit_log2(match_limit),
                }
            })
            .collect();
        let mut result = SchedulerSnapshot {
            scheduler: "backoff",
            rules,
            ..SchedulerSnapshot::default()
        };
        result.recompute_aggregates();
        result
    }

    fn can_stop(&mut self, iteration: usize) -> bool {
        let n_stats = self.stats.len();

        let mut banned: Vec<_> = self
            .stats
            .iter_mut()
            .filter(|(_, s)| s.banned_until > iteration)
            .collect();

        if banned.is_empty() {
            true
        } else {
            let min_ban = banned
                .iter()
                .map(|(_, s)| s.banned_until)
                .min()
                .expect("banned cannot be empty here");

            assert!(min_ban >= iteration);
            let delta = min_ban - iteration;

            let mut unbanned = vec![];
            for (name, s) in &mut banned {
                s.banned_until -= delta;
                if s.banned_until == iteration {
                    unbanned.push(name.as_str());
                }
            }

            assert!(!unbanned.is_empty());
            info!(
                "Banned {}/{}, fast-forwarded by {} to unban {}",
                banned.len(),
                n_stats,
                delta,
                unbanned.join(", "),
            );

            false
        }
    }

    fn search_rewrite<'a>(
        &mut self,
        iteration: usize,
        egraph: &EGraph<L, N>,
        rewrite: &'a Rewrite<L, N>,
    ) -> Vec<SearchMatches<'a, L>> {
        let stats = self.rule_stats(rewrite.name);

        if iteration < stats.banned_until {
            debug!(
                "Skipping {} ({}-{}), banned until {}...",
                rewrite.name, stats.times_applied, stats.times_banned, stats.banned_until,
            );
            return vec![];
        }

        let threshold = effective_match_limit(stats.match_limit, stats.times_banned);
        let matches = rewrite.search_with_limit(egraph, threshold.saturating_add(1));
        let total_len: usize = matches.iter().map(|m| m.substs.len()).sum();
        if total_len > threshold {
            let ban_length = effective_match_limit(stats.ban_length, stats.times_banned);
            stats.times_banned += 1;
            stats.banned_until = iteration.saturating_add(ban_length);
            info!(
                "Banning {} ({}-{}) for {} iters: {} < {}",
                rewrite.name,
                stats.times_applied,
                stats.times_banned,
                ban_length,
                threshold,
                total_len,
            );
            vec![]
        } else {
            stats.times_applied += 1;
            matches
        }
    }
}

/// Custom data to inject into the [`Iteration`]s recorded by a [`Runner`]
///
/// This trait allows you to add custom data to the [`Iteration`]s
/// recorded as a [`Runner`] applies rules.
///
/// See the [`Runner`] docs for an example.
///
/// [`Runner`] is generic over the [`IterationData`] that it will be in the
/// [`Iteration`]s, but by default it uses `()`.
///
pub trait IterationData<L, N>: Sized
where
    L: Language,
    N: Analysis<L>,
{
    /// Given the current [`Runner`], make the
    /// data to be put in this [`Iteration`].
    fn make(runner: &Runner<L, N, Self>) -> Self;
}

impl<L, N> IterationData<L, N> for ()
where
    L: Language,
    N: Analysis<L>,
{
    fn make(_: &Runner<L, N, Self>) -> Self {}
}

#[cfg(test)]
mod scheduler_snapshot_tests {
    use super::*;
    use crate::{SymbolLang, rewrite as rw};

    fn snapshot(
        scheduler: &BackoffScheduler,
        iteration: usize,
        rules: &[&Rewrite<SymbolLang, ()>],
    ) -> SchedulerSnapshot {
        <BackoffScheduler as RewriteScheduler<SymbolLang, ()>>::snapshot(
            scheduler, iteration, rules,
        )
    }

    #[test]
    fn snapshot_includes_every_rule_before_first_search() {
        let a = rw!("a"; "?x" => "(f ?x)");
        let b = rw!("b"; "?x" => "(g ?x)");
        let rules = [&a, &b];
        let state = snapshot(&BackoffScheduler::default(), 0, &rules);
        assert_eq!(state.scheduler, "backoff");
        assert_eq!(state.n_active, 2);
        assert_eq!(
            state.rules.iter().map(|rule| rule.name).collect::<Vec<_>>(),
            vec![Symbol::from("a"), Symbol::from("b")]
        );
        assert!(state.rules.iter().all(|rule| rule.will_search));
        assert!(state.rules.iter().all(|rule| !rule.newly_unbanned));
    }

    #[test]
    fn active_banned_and_newly_unbanned_state_is_exact() {
        let a = rw!("a"; "?x" => "(f ?x)");
        let b = rw!("b"; "?x" => "(g ?x)");
        let c = rw!("c"; "?x" => "(h ?x)");
        let rules = [&a, &b, &c];
        let mut scheduler = BackoffScheduler::default();
        scheduler.stats.insert(
            "a".into(),
            RuleStats {
                times_applied: 0,
                banned_until: 8,
                times_banned: 2,
                match_limit: 10,
                ban_length: 3,
            },
        );
        scheduler.stats.insert(
            "b".into(),
            RuleStats {
                times_applied: 0,
                banned_until: 6,
                times_banned: 1,
                match_limit: 10,
                ban_length: 3,
            },
        );
        scheduler.stats.insert(
            "c".into(),
            RuleStats {
                times_applied: 0,
                banned_until: 0,
                times_banned: 4,
                match_limit: 10,
                ban_length: 3,
            },
        );

        let state = snapshot(&scheduler, 6, &rules);
        assert_eq!(state.n_active, 2);
        assert_eq!(state.n_banned, 1);
        assert_eq!(state.n_newly_unbanned, 1);
        assert_eq!(state.min_ban_remaining, 2);
        assert_eq!(state.total_times_banned, 7);
        assert!(!state.rules[0].will_search);
        assert_eq!(state.rules[0].ban_remaining, 2);
        assert!(state.rules[1].will_search);
        assert!(state.rules[1].newly_unbanned);
        assert!(state.rules[2].will_search);
    }

    #[test]
    fn effective_limit_saturates_instead_of_shifting_with_overflow() {
        assert_eq!(effective_match_limit(1_000, 3), 8_000);
        assert_eq!(effective_match_limit(usize::MAX, 1), usize::MAX);
        assert_eq!(effective_match_limit(1, usize::MAX), usize::MAX);
        assert_eq!(effective_match_limit(0, usize::MAX), 0);
        assert!(match_limit_log2(effective_match_limit(1, usize::MAX)).is_finite());
    }

    #[test]
    fn introspected_effective_limit_is_the_search_threshold() {
        let rewrite = rw!("explode"; "?x" => "(f ?x)");
        let rules = [&rewrite];
        let mut scheduler = BackoffScheduler::default().with_initial_match_limit(0);
        let before = snapshot(&scheduler, 0, &rules);
        assert_eq!(before.rules[0].match_limit, 0);
        assert_eq!(before.rules[0].log2_match_limit, 0.0);

        let mut egraph = EGraph::<SymbolLang, ()>::default();
        egraph.add_expr(&"x".parse().unwrap());
        egraph.rebuild();
        let matches = <BackoffScheduler as RewriteScheduler<SymbolLang, ()>>::search_rewrite(
            &mut scheduler,
            0,
            &egraph,
            &rewrite,
        );
        assert!(matches.is_empty());
        assert_eq!(scheduler.stats[&Symbol::from("explode")].times_banned, 1);
    }

    #[derive(Debug, PartialEq)]
    struct CapturedSnapshot(SchedulerSnapshot);

    impl IterationData<SymbolLang, ()> for CapturedSnapshot {
        fn make(runner: &Runner<SymbolLang, (), Self>) -> Self {
            Self(runner.scheduler_snapshot.clone())
        }
    }

    #[test]
    fn runner_stores_snapshot_from_before_search() {
        let rewrite = rw!("expand"; "?a" => "(f ?a)");
        let runner = Runner::<SymbolLang, (), CapturedSnapshot>::new(())
            .with_expr(&"x".parse().unwrap())
            .with_iter_limit(3)
            .with_scheduler(
                BackoffScheduler::default()
                    .with_initial_match_limit(0)
                    .with_ban_length(5),
            )
            .run(&[rewrite]);

        assert_eq!(runner.iterations[0].data.0.total_times_banned, 0);
        assert_eq!(runner.iterations[0].data.0.rules.len(), 1);
        assert_eq!(runner.iterations[1].data.0.total_times_banned, 1);
        assert!(runner.iterations[1].data.0.rules[0].will_search);
        assert!(!runner.iterations[1].data.0.rules[0].newly_unbanned);
        assert_eq!(runner.iterations[2].data.0.total_times_banned, 2);
    }

    #[test]
    fn fast_forwarded_rule_is_active_in_the_next_snapshot() {
        let rewrite = rw!("a"; "?x" => "(f ?x)");
        let rules = [&rewrite];
        let mut scheduler = BackoffScheduler::default();
        scheduler.stats.insert(
            "a".into(),
            RuleStats {
                times_applied: 0,
                banned_until: 10,
                times_banned: 1,
                match_limit: 10,
                ban_length: 3,
            },
        );
        assert!(
            !<BackoffScheduler as RewriteScheduler<SymbolLang, ()>>::can_stop(&mut scheduler, 4)
        );
        let state = snapshot(&scheduler, 5, &rules);
        assert!(state.rules[0].will_search);
        assert!(!state.rules[0].newly_unbanned);
        assert_eq!(state.rules[0].ban_remaining, 0);
    }
}

#[cfg(test)]
mod memory_peak_tests {
    use super::*;
    use crate::{SymbolLang, rewrite as rw};
    use std::sync::Mutex;

    static SAMPLES: Mutex<Vec<u64>> = Mutex::new(Vec::new());
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    fn sample() -> u64 {
        SAMPLES.lock().unwrap().remove(0)
    }

    #[derive(Debug)]
    struct CapturedPeak {
        peak: IterationMemoryPeak,
        end: u64,
    }

    impl IterationData<SymbolLang, ()> for CapturedPeak {
        fn make(runner: &Runner<SymbolLang, (), Self>) -> Self {
            Self {
                peak: runner.iteration_memory_peak().unwrap(),
                end: runner.memory_reading().unwrap(),
            }
        }
    }

    #[test]
    fn transient_search_peak_is_attributed_retained_and_reset() {
        let _guard = TEST_LOCK.lock().unwrap();
        // tracker initialization; iteration 0 start/search/apply/finalize;
        // iteration 1 start/search/apply/finalize; final run report.
        *SAMPLES.lock().unwrap() = vec![90, 100, 500, 120, 110, 200, 180, 170, 160, 150];
        let rewrite = rw!("expand"; "?a" => "(f ?a)");
        let runner =
            Runner::<SymbolLang, (), CapturedPeak>::new_with_memory_tracker((), sample, None)
                .with_expr(&"x".parse().unwrap())
                .with_iter_limit(2)
                .run(&[rewrite]);

        let first = &runner.iterations[0].data;
        assert_eq!(first.peak.iteration_start_allocated, 100);
        assert_eq!(first.peak.iteration_peak_allocated, 500);
        assert_eq!(first.peak.peak_phase, MemorySamplePhase::AfterRuleSearch);
        assert_eq!(first.peak.peak_rule, Some(Symbol::from("expand")));
        assert_eq!(first.end, 110);

        let second = &runner.iterations[1].data;
        assert_eq!(second.peak.iteration_start_allocated, 200);
        assert_eq!(second.peak.iteration_peak_allocated, 200);
        assert_eq!(second.peak.peak_phase, MemorySamplePhase::BeforeHooks);
        assert_eq!(second.peak.peak_rule, None);
        assert_eq!(second.end, 160);
    }

    #[test]
    fn transient_crossing_survives_after_match_vector_is_dropped() {
        let _guard = TEST_LOCK.lock().unwrap();
        // No application sample occurs because the post-search hard-limit
        // check stops the iteration and drops its match vector.
        *SAMPLES.lock().unwrap() = vec![90, 100, 500, 120, 110];
        let rewrite = rw!("catastrophic"; "?a" => "(f ?a)");
        let runner =
            Runner::<SymbolLang, (), CapturedPeak>::new_with_memory_tracker((), sample, Some(400))
                .with_expr(&"x".parse().unwrap())
                .with_iter_limit(5)
                .run(&[rewrite]);

        assert!(matches!(
            runner.stop_reason,
            Some(StopReason::MemoryLimit(500))
        ));
        let data = &runner.iterations[0].data;
        assert_eq!(data.peak.iteration_peak_allocated, 500);
        assert_eq!(data.peak.peak_rule, Some(Symbol::from("catastrophic")));
        assert_eq!(data.end, 120);
        assert!(data.end < 400);
    }
}
