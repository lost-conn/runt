//! Coalescing param edits (DESIGN §10, §6).
//!
//! Dragging a slider produces an edit per pixel of travel. Each one would
//! regenerate a mesh — cheap through the cache on a *revisited* value, free
//! never. So edits are held for a moment and only the last one is sent.
//!
//! This is not throttling. A throttle sends the first edit and drops the rest,
//! which makes a drag feel laggy and, worse, can end on a value that was never
//! sent. A debounce always sends the **latest** value, and always sends
//! *something* after the drag stops — [`take_expired`] guarantees the terminal
//! edit lands.
//!
//! Time is passed in rather than read, for the same reason the engine never
//! reads a clock (DESIGN §4): it makes the whole thing testable without
//! sleeping.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

/// The delay DESIGN §10 asks for: long enough to swallow a drag, short enough
/// that letting go feels immediate.
pub const DEFAULT_DELAY: Duration = Duration::from_millis(120);

/// Latest-value-wins coalescing, keyed by `K`.
///
/// Edits to *different* keys never suppress each other — dragging the radius
/// while the seed field still has a pending change must not lose the seed — so
/// each key carries its own deadline.
///
/// `BTreeMap` rather than `HashMap`: the order pending edits are flushed in is
/// then a property of the keys, not of a hash seed, which is the same rule
/// DESIGN §3 puts on the sim.
#[derive(Debug)]
pub struct Debouncer<K: Ord, V> {
    delay: Duration,
    pending: BTreeMap<K, (V, Instant)>,
}

impl<K: Ord, V> Default for Debouncer<K, V> {
    fn default() -> Debouncer<K, V> {
        Debouncer::new(DEFAULT_DELAY)
    }
}

impl<K: Ord, V> Debouncer<K, V> {
    pub fn new(delay: Duration) -> Debouncer<K, V> {
        Debouncer {
            delay,
            pending: BTreeMap::new(),
        }
    }

    pub fn delay(&self) -> Duration {
        self.delay
    }

    /// Record an edit, replacing any pending edit for the same key and pushing
    /// that key's deadline out.
    pub fn push(&mut self, key: K, value: V, now: Instant) {
        self.pending.insert(key, (value, now + self.delay));
    }

    /// Every edit whose quiet period has elapsed, removed from the queue.
    ///
    /// Returned in key order, so a caller draining several generators at once
    /// applies them in a defined sequence.
    pub fn take_expired(&mut self, now: Instant) -> Vec<(K, V)>
    where
        K: Clone,
    {
        let ready: Vec<K> = self
            .pending
            .iter()
            .filter(|(_, (_, deadline))| *deadline <= now)
            .map(|(k, _)| k.clone())
            .collect();
        ready
            .into_iter()
            .map(|k| {
                let (v, _) = self.pending.remove(&k).expect("just listed");
                (k, v)
            })
            .collect()
    }

    /// Drain everything regardless of deadline — what a "Save" button does
    /// before writing, so a half-second-old slider drag is not lost to the file.
    pub fn flush(&mut self) -> Vec<(K, V)> {
        std::mem::take(&mut self.pending)
            .into_iter()
            .map(|(k, (v, _))| (k, v))
            .collect()
    }

    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    pub fn len(&self) -> usize {
        self.pending.len()
    }

    /// How long until the next edit is due, or `None` if nothing is pending.
    /// A UI can use this to decide how long to wait before looking again.
    pub fn time_until_next(&self, now: Instant) -> Option<Duration> {
        self.pending
            .values()
            .map(|(_, deadline)| deadline.saturating_duration_since(now))
            .min()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const D: Duration = Duration::from_millis(100);

    #[test]
    fn nothing_is_released_before_the_delay() {
        let t0 = Instant::now();
        let mut d: Debouncer<u32, i32> = Debouncer::new(D);
        d.push(1, 10, t0);
        assert!(d.take_expired(t0).is_empty());
        assert!(d.take_expired(t0 + Duration::from_millis(99)).is_empty());
        assert_eq!(d.take_expired(t0 + D), vec![(1, 10)]);
    }

    /// The property that matters: a drag sends one edit, and it is the *last*
    /// value, not the first.
    #[test]
    fn a_drag_collapses_to_its_final_value() {
        let t0 = Instant::now();
        let mut d: Debouncer<u32, i32> = Debouncer::new(D);
        for (i, value) in (0..40).enumerate() {
            // One edit every 5 ms — a plausible drag.
            d.push(1, value, t0 + Duration::from_millis(5 * i as u64));
        }
        // Still nothing, because every push moved the deadline.
        assert!(d.take_expired(t0 + Duration::from_millis(200)).is_empty());
        let released = d.take_expired(t0 + Duration::from_millis(295));
        assert_eq!(released, vec![(1, 39)], "the terminal value must be the one sent");
        assert!(d.is_empty());
    }

    #[test]
    fn separate_keys_have_separate_deadlines() {
        let t0 = Instant::now();
        let mut d: Debouncer<&str, i32> = Debouncer::new(D);
        d.push("radius", 1, t0);
        d.push("seed", 2, t0 + Duration::from_millis(50));

        assert_eq!(d.take_expired(t0 + D), vec![("radius", 1)]);
        assert_eq!(d.len(), 1, "the seed edit is still waiting its own turn");
        assert_eq!(
            d.take_expired(t0 + Duration::from_millis(150)),
            vec![("seed", 2)]
        );
    }

    #[test]
    fn expired_edits_come_out_in_key_order() {
        let t0 = Instant::now();
        let mut d: Debouncer<u32, &str> = Debouncer::new(D);
        d.push(3, "c", t0);
        d.push(1, "a", t0);
        d.push(2, "b", t0);
        assert_eq!(
            d.take_expired(t0 + D),
            vec![(1, "a"), (2, "b"), (3, "c")]
        );
    }

    #[test]
    fn flush_ignores_deadlines() {
        let t0 = Instant::now();
        let mut d: Debouncer<u32, i32> = Debouncer::new(D);
        d.push(1, 10, t0);
        d.push(2, 20, t0);
        assert_eq!(d.flush(), vec![(1, 10), (2, 20)]);
        assert!(d.is_empty());
    }

    #[test]
    fn the_next_deadline_is_the_earliest_one() {
        let t0 = Instant::now();
        let mut d: Debouncer<u32, i32> = Debouncer::new(D);
        assert_eq!(d.time_until_next(t0), None);
        d.push(1, 0, t0 + Duration::from_millis(50));
        d.push(2, 0, t0);
        assert_eq!(d.time_until_next(t0), Some(D));
        // Never negative, so a caller can hand it straight to a sleep.
        assert_eq!(d.time_until_next(t0 + Duration::from_secs(5)), Some(Duration::ZERO));
    }
}
