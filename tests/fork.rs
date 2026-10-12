//! A process forked after `leiden` ran (R's mclapply, Python's multiprocessing with fork) must not
//! reuse the parent's cached thread pool: its worker threads do not exist in the child, so it hung.
//! Own test binary: fork() with other test threads running is unsafe.
#![cfg(unix)]

use clust_libreville_rs::{leiden, Options};

mod common;

#[test]
fn leiden_runs_in_a_forked_child() {
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
