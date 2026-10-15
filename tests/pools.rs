//! The per-thread pool cache (`with_pool`): a forked child must not reuse the parent's pools, and
//! `with_threads` around `leiden` must not nest a second pool inside the first. Own test binary,
//! one test at a time (fork() with other test threads running is unsafe; the thread count is
//! process-wide).
#![cfg(unix)]

use std::sync::Mutex;

use clust_libreville_rs::{leiden, Options};

mod common;

static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

/// A process forked after `leiden` ran (R's mclapply, Python's multiprocessing with fork) must not
/// reuse the parent's cached thread pool: its worker threads do not exist in the child, so it hung.
#[test]
fn leiden_runs_in_a_forked_child() {
    let _lock = ONE_AT_A_TIME.lock().unwrap();
    let (n, edges) = common::planted(3, 4, 30, 0.3, 0.02);
    let g = common::graph(n, &edges);
    let o = Options { threads: Some(2), ..Default::default() };
    let parent = leiden(&g, &o); // builds and caches the 2-thread pool
    match unsafe { libc::fork() } {
        0 => {
            unsafe { libc::alarm(30) }; // a hang ends in SIGALRM instead of a stuck test
            let ok = leiden(&g, &o) == parent && leiden(&g, &Options { threads: Some(3), ..o }) == parent;
            unsafe { libc::_exit(if ok { 0 } else { 1 }) }
        }
        pid => {
            assert!(pid > 0, "fork failed");
            let mut status = 0;
            assert_eq!(unsafe { libc::waitpid(pid, &mut status, 0) }, pid);
            assert!(libc::WIFEXITED(status), "child killed by signal {} (hung: SIGALRM = 14)", libc::WTERMSIG(status));
            assert_eq!(libc::WEXITSTATUS(status), 0, "child got different labels");
        }
    }
}

/// `with_threads(t, || leiden(.., threads: t))`, as the bindings call it, runs the inner call in
/// the outer pool: t new OS threads on the first call, none afterwards. (It used to build a second
/// t-thread pool on the outer pool's worker, nested and idle.)
#[test]
#[cfg(target_os = "linux")]
fn with_threads_does_not_nest_a_second_pool() {
    let _lock = ONE_AT_A_TIME.lock().unwrap();
    let threads = || -> usize {
        let s = std::fs::read_to_string("/proc/self/status").unwrap();
        s.lines().find_map(|l| l.strip_prefix("Threads:")).unwrap().trim().parse().unwrap()
    };
    let (n, edges) = common::planted(3, 4, 30, 0.3, 0.02);
    let g = common::graph(n, &edges);
    let o = Options { threads: Some(5), ..Default::default() };
    let want = clust_libreville_rs::with_threads(Some(5), || leiden(&g, &o));
    let before = threads();
    for _ in 0..3 {
        assert_eq!(clust_libreville_rs::with_threads(Some(5), || leiden(&g, &o)), want);
    }
    assert_eq!(threads(), before, "repeated calls spawned threads");
    let o = Options { threads: Some(7), ..Default::default() };
    clust_libreville_rs::with_threads(Some(7), || leiden(&g, &o));
    assert_eq!(threads(), before + 7, "a new size adds exactly its pool");
}
