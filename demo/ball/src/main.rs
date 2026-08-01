//! `runt-ball` — the native binary (DESIGN §12 step 6).
//!
//! ```text
//! runt-ball                     play
//! runt-ball --record run.trace  play, and write the input trace on exit
//! runt-ball --replay run.trace  watch a recorded run play itself back
//! ```
//!
//! A trace is `(tick index, input event)` pairs in postcard — DESIGN §4's
//! "replays are just recorded input traces + seeds", with the seed already in
//! `level1.ron`. Both flags may be given at once, which re-records a replay: the
//! recorder sits downstream of the player in the tick, so the output file must
//! come back byte-identical, and that is a cheap end-to-end check on the whole
//! mechanism.
//!
//! The flags are native-only. On the web there is no argv and no file system to
//! write to, and the wasm entry point in `lib.rs` simply plays.

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    if let Err(e) = native::run() {
        eprintln!("runt-ball: {e}\n{}", native::USAGE);
        std::process::exit(2);
    }
}

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use std::path::PathBuf;

    use runt_ball::config;
    use runt_core::trace::InputTrace;

    pub const USAGE: &str = "usage: runt-ball [--record <file>] [--replay <file>]";

    /// `(--record path, --replay path)`.
    type Flags = (Option<PathBuf>, Option<PathBuf>);

    pub fn run() -> Result<(), String> {
        let Some((record, replay)) = parse_args(std::env::args().skip(1))? else {
            println!("{USAGE}");
            return Ok(());
        };

        let mut config = config();

        if let Some(path) = &replay {
            let bytes = std::fs::read(path).map_err(|e| format!("reading {path:?}: {e}"))?;
            let trace = InputTrace::from_bytes(&bytes).map_err(|e| format!("{path:?}: {e}"))?;
            log::info!(
                "replaying {path:?}: {} events over {} ticks",
                trace.len(),
                trace.last_tick().map_or(0, |t| t + 1)
            );
            // Chained onto whatever `config()` already set up, so the game's own
            // systems are installed either way.
            let previous = config.setup.take();
            config = config.with_setup(move |sim| {
                if let Some(previous) = previous {
                    previous(sim);
                }
                sim.play_input_trace(trace);
            });
        }

        if let Some(path) = record.clone() {
            let previous = config.setup.take();
            config = config
                .with_setup(move |sim| {
                    if let Some(previous) = previous {
                        previous(sim);
                    }
                    sim.record_input_trace();
                })
                .with_on_exit(move |sim| {
                    let Some(trace) = sim.input_trace() else {
                        return;
                    };
                    match trace.to_bytes().map_err(|e| e.to_string()).and_then(|bytes| {
                        std::fs::write(&path, &bytes)
                            .map(|()| bytes.len())
                            .map_err(|e| e.to_string())
                    }) {
                        Ok(len) => log::info!(
                            "recorded {} events over {} ticks → {path:?} ({len} bytes)",
                            trace.len(),
                            sim.tick_count()
                        ),
                        Err(e) => log::error!("could not write {path:?}: {e}"),
                    }
                });
        }

        runt_app::run_with(config);
        Ok(())
    }

    /// `--record <file>` / `--replay <file>`, and nothing else. Hand-rolled
    /// because a clap dependency for two flags is not a trade this crate wants.
    ///
    /// `Ok(None)` means "the user asked for help, we printed it, stop".
    fn parse_args(args: impl Iterator<Item = String>) -> Result<Option<Flags>, String> {
        let mut record = None;
        let mut replay = None;
        let mut args = args.peekable();
        while let Some(arg) = args.next() {
            let slot = match arg.as_str() {
                "--record" => &mut record,
                "--replay" => &mut replay,
                "-h" | "--help" => return Ok(None),
                other => return Err(format!("unknown argument {other:?}")),
            };
            let path = args
                .next()
                .ok_or_else(|| format!("{arg} needs a file path"))?;
            if slot.replace(PathBuf::from(path)).is_some() {
                return Err(format!("{arg} given twice"));
            }
        }
        Ok(Some((record, replay)))
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn parse(args: &[&str]) -> Result<Option<Flags>, String> {
            parse_args(args.iter().map(|s| s.to_string()))
        }

        #[test]
        fn the_flags_parse_and_the_bad_ones_do_not() {
            assert_eq!(parse(&[]).unwrap(), Some((None, None)));
            assert_eq!(
                parse(&["--replay", "run.trace"]).unwrap(),
                Some((None, Some(PathBuf::from("run.trace"))))
            );
            assert_eq!(
                parse(&["--record", "a", "--replay", "b"]).unwrap(),
                Some((Some(PathBuf::from("a")), Some(PathBuf::from("b")))),
                "recording a replay is legal, and is a self-check"
            );
            assert_eq!(parse(&["--help"]).unwrap(), None);
            assert!(parse(&["--record"]).is_err(), "a flag needs its path");
            assert!(parse(&["--nope", "x"]).is_err());
            assert!(parse(&["--record", "a", "--record", "b"]).is_err());
        }
    }
}

#[cfg(target_arch = "wasm32")]
fn main() {}
