//! The game's sound (DESIGN §8's phase-3 item 3, in the demo).
//!
//! Everything here is *game* code: which sounds exist, what they are made of,
//! and when they play. `runt-core` supplies a queue and a pan helper; nothing in
//! the engine knows this file exists, and nothing in this file knows whether it
//! is talking to a cpal stream or an `AudioWorkletProcessor` (DESIGN §8: "the
//! engine cannot tell them apart").
//!
//! ```text
//! bank()                     four presets, as params — the content (§6, §8)
//! GameAudio                  the game's audio state: listener, sequencer, tracking
//! game_audio (FixedSim)      one system, after follow_camera, before flush_audio
//! ```
//!
//! ## Why one system, and why it sits after the camera
//!
//! Pan and distance are computed against the camera pose, and `follow_camera`
//! moves the camera late in the tick. A sound emitted before it would be placed
//! against the *previous* tick's viewpoint — 16 ms of lag, most visible exactly
//! when the camera is swinging, which is exactly when a pickup is being taken.
//! So the game's rules run where they always did (before the camera, so a
//! kill-plane reset teleports the ball before the camera lunges at the old spot)
//! and the *audio* is emitted afterwards, from state the rules left behind.
//!
//! [`collect_pickups`](crate::game::collect_pickups) therefore records what it
//! took rather than playing it. That is the one line of coupling between a rule
//! and a sound, and it buys the correct listener pose for every positional
//! event.
//!
//! ## Determinism
//!
//! No clock, no RNG, no hash iteration. The pickup seed is the score *before*
//! the pickup landed, so the notes walk up the scale in collection order; the
//! fanfare is scheduled on tick offsets from the tick the win happened on. Both
//! are pure functions of the tick stream, which is what makes
//! `tests/audio.rs`'s "replay produces the same events on the same ticks" hold.

use bevy_ecs::prelude::*;
use glam::Vec3;

use runt_audio::{DroneParams, PatchBank, PatchDef, PluckParams};
use runt_core::audio::{AudioOut, Listener, ParamId, PatchId, Rolloff, VoiceId};
use runt_core::ecs::TickCount;
use runt_core::physics::{Ball, Grounded, Velocity};
use runt_core::{camera::Camera, Input, Transform};

use crate::game::{GameState, Phase};

// ---------------------------------------------------------------------------
// The bank — audio params are content (DESIGN §8, §6)
// ---------------------------------------------------------------------------

/// A ring collected: a bright ping. The seed walks it up a pentatonic scale, so
/// twelve rings are a phrase rather than the same beep twelve times.
pub const PICKUP: PatchId = PatchId::new("pickup");
/// The ball landing hard: the same synthesis model an octave and a half down,
/// with the filter almost shut. A "pluck" with no top end is a thud.
pub const THUD: PatchId = PatchId::new("thud");
/// One note of the win fanfare. Pitched per note with
/// [`ParamId::PITCH`](runt_core::audio::ParamId::PITCH).
pub const CHIME: PatchId = PatchId::new("chime");
/// The bed. Very quiet, very slow, and the first thing to delete if it ever
/// fights the effects.
pub const AMBIENCE: PatchId = PatchId::new("ambience");

/// The four presets, as params.
///
/// This is the shape DESIGN §8 asks for — *"a `Serialize + Hash` param struct
/// plus an explicit seed"* — and it is deliberately data rather than code: phase
/// 4's editor panels edit exactly these fields, and a scene RON will carry them
/// without anything in this file changing.
pub fn bank() -> PatchBank {
    PatchBank::new()
        .with(
            "pickup",
            PatchDef::Pluck(PluckParams {
                base_hz: 523.25, // C5
                // Major pentatonic over an octave and a half: collected in any
                // order, in any combination, it stays consonant.
                steps: vec![0, 2, 4, 7, 9, 12, 14, 16, 19],
                detune: 1.004,
                detune_gain: 0.45,
                attack_s: 0.003,
                decay_s: 0.28,
                cutoff_hz: 1400.0,
                cutoff_env: 4.0,
                resonance: 1.1,
                gain: 0.5,
                jitter_semitones: 0.03,
            }),
        )
        .with(
            "thud",
            PatchDef::Pluck(PluckParams {
                base_hz: 62.0, // ~B1
                steps: vec![0, -2],
                detune: 1.012, // wide, so it beats rather than rings
                detune_gain: 0.8,
                attack_s: 0.006,
                decay_s: 0.22,
                cutoff_hz: 180.0,
                cutoff_env: 2.2,
                resonance: 0.8,
                gain: 0.9,
                jitter_semitones: 0.5,
            }),
        )
        .with(
            "chime",
            PatchDef::Pluck(PluckParams {
                base_hz: 659.25, // E5, the root of the fanfare
                steps: vec![0],  // pitched by SetParam instead — see `fanfare`
                detune: 1.002,
                detune_gain: 0.35,
                attack_s: 0.004,
                decay_s: 0.9,
                cutoff_hz: 2600.0,
                cutoff_env: 2.5,
                resonance: 0.9,
                gain: 0.55,
                jitter_semitones: 0.0,
            }),
        )
        .with(
            "ambience",
            PatchDef::Drone(DroneParams {
                base_hz: 55.0, // A1, two octaves under the pickup root
                detune: 1.005,
                sub_gain: 0.5,
                cutoff_hz: 260.0,
                resonance: 1.2,
                lfo_hz: 0.07, // one sweep every fourteen seconds
                lfo_depth: 0.45,
                attack_s: 4.0,
                release_s: 2.0,
                gain: 0.16,
                jitter_semitones: 0.0,
            }),
        )
}

// ---------------------------------------------------------------------------
// Tuning
// ---------------------------------------------------------------------------

/// How the demo's camera hears the world.
///
/// The follow camera sits ~10 m back, so a 7 m reference radius means anything
/// near the ball is at full level and the far corner of the patch is audibly
/// distant without disappearing. `pan_width` above 1 spreads the field slightly
/// wider than the screen, which reads better on headphones for a camera that is
/// behind the player rather than at their eyes.
pub const ROLLOFF: Rolloff = Rolloff {
    reference: 7.0,
    pan_width: 1.2,
};

/// Impact speed, m/s, below which a landing is not worth a sound. Above the
/// speed a ball reaches rolling over ordinary terrain, so ordinary rolling is
/// silent and only a real drop is heard.
pub const LANDING_SPEED: f32 = 4.0;

/// Impact speed at which the thud reaches full volume. Above it the gain is
/// capped, so falling off the map is not a jump-scare.
pub const LANDING_FULL_SPEED: f32 = 14.0;

/// Loudest a landing may be.
pub const LANDING_GAIN: f32 = 0.85;

/// Level a pickup ping is played at before the distance law.
pub const PICKUP_GAIN: f32 = 0.9;

/// Ambience level. Low enough that it reads as room tone under the effects
/// rather than as music competing with them.
pub const AMBIENCE_GAIN: f32 = 0.9;

/// Ticks after the player's first input before the ambience starts.
///
/// ## Why the bed waits for a keystroke
///
/// A browser will not start an `AudioContext` until the user has interacted with
/// the page, and until it does, the host **drops** what it is handed — the right
/// behaviour for one-shots (nobody wants thirty seconds of queued pickup
/// noises to fire on their first click) and the wrong behaviour for a
/// once-per-run sound that would then never play at all.
///
/// So the game starts its bed on the first tick that carries player input, half
/// a second late. That half second is slack for `AudioContext.resume()`, which
/// is a promise and does not necessarily resolve before the next frame's submit.
///
/// It is worth being clear about what this is *not*: the engine has no idea a
/// browser exists, and neither does this constant. "Start the ambience once the
/// player is actually playing" is an ordinary game decision that happens to line
/// up with the platform's rule — and, crucially, it is a pure function of the
/// input trace, so a replay starts the bed on the same tick as the run it came
/// from. Native has no such policy and behaves identically.
pub const AMBIENCE_DELAY_TICKS: u64 = 30;

/// Ticks between the fanfare's notes: 8/60 s ≈ 133 ms, a brisk arpeggio.
pub const FANFARE_SPACING: u64 = 8;

/// The fanfare, as pitch multipliers on the `chime` preset's root — a major
/// triad, played by three `Play`s each followed immediately by a
/// [`ParamId::PITCH`](runt_core::audio::ParamId::PITCH) edit.
///
/// Doing it with one preset and three edits rather than three presets is the
/// point of having `SetParam` at all: a game that wants an arbitrary pitch does
/// not have to mint content for it.
pub const FANFARE: [f32; 3] = [1.0, 1.25, 1.5];

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

/// Everything the game's audio remembers between ticks.
///
/// A resource rather than components because there is exactly one listener, one
/// ambience voice and one fanfare. Sim state in the sense that it lives in the
/// world and is written inside the tick — but *derived* state: nothing outside
/// this module reads it, and deleting the whole module changes no trajectory.
#[derive(Resource, Clone, Debug, Default, PartialEq)]
pub struct GameAudio {
    /// Rings taken this tick: `(world position, how many had been taken before
    /// it)`. Written by [`collect_pickups`](crate::game::collect_pickups),
    /// drained by [`game_audio`].
    pub collected: Vec<(Vec3, u32)>,
    /// The ambience voice, once started.
    pub ambience: Option<VoiceId>,
    /// The first tick that carried player input. See [`AMBIENCE_DELAY_TICKS`].
    pub first_input_tick: Option<u64>,
    /// The tick the win happened on, if it has.
    pub fanfare_start: Option<u64>,
    /// Notes of the fanfare already played.
    pub fanfare_played: usize,
    /// Ball contact state as of the previous tick, for the landing edge.
    pub was_grounded: bool,
    /// The ball's velocity at the end of the previous tick — the closest thing
    /// to an impact speed that exists after the contact solve has already
    /// absorbed it.
    pub prev_velocity: Vec3,
    /// The ball's position at the end of the previous tick, for spotting a
    /// teleport (see [`game_audio`]).
    pub prev_position: Option<Vec3>,
}

// ---------------------------------------------------------------------------
// The system
// ---------------------------------------------------------------------------

/// `FixedSim`, after `follow_camera` and before `flush_audio`: turn this tick's
/// game events into audio events.
///
/// Four jobs, in a fixed order so the batch a tick emits is a function of the
/// tick and not of anything else:
///
/// 1. start the ambience, once, on the first tick;
/// 2. one ping per ring taken, placed where the ring was;
/// 3. a thud on the airborne→grounded edge, if it was hard enough;
/// 4. the next note of the win fanfare, if one is due.
#[allow(clippy::type_complexity)] // Bevy system params read worse behind aliases.
pub fn game_audio(
    tick: Res<TickCount>,
    state: Res<GameState>,
    mut audio: ResMut<GameAudio>,
    mut out: ResMut<AudioOut>,
    input: Res<Input>,
    cameras: Query<&Transform, (With<Camera>, Without<Ball>)>,
    balls: Query<(&Transform, &Velocity, Option<&Grounded>), With<Ball>>,
) {
    // The listener is the camera. DESIGN §5 has exactly one per render; if a
    // scene somehow has none, everything falls back to centre and full gain,
    // which is the right degradation for a HUD-less sound.
    let listener = cameras
        .iter()
        .next()
        .map(Listener::from_pose)
        .unwrap_or_default();

    // -- 1. ambience -------------------------------------------------------
    if audio.first_input_tick.is_none() && player_is_here(&input) {
        audio.first_input_tick = Some(tick.0);
    }
    if audio.ambience.is_none()
        && audio
            .first_input_tick
            .is_some_and(|at| tick.0 >= at + AMBIENCE_DELAY_TICKS)
    {
        let voice = out.play(AMBIENCE, 0, AMBIENCE_GAIN, 0.0);
        audio.ambience = Some(voice);
    }

    // -- 2. rings ----------------------------------------------------------
    //
    // The seed is the number of rings already taken, so the scale walks upward
    // as the run progresses — a progress cue that costs one integer and no
    // engine feature.
    for (position, ordinal) in std::mem::take(&mut audio.collected) {
        out.play_at(
            PICKUP,
            ordinal as u64,
            PICKUP_GAIN,
            position,
            &listener,
            ROLLOFF,
        );
    }

    // -- 3. landing --------------------------------------------------------
    if let Ok((transform, velocity, contact)) = balls.get(state.player) {
        let position = transform.translation;
        let grounded = contact.is_some_and(|c| c.grounded);
        let normal = contact.map_or(Vec3::Y, |c| c.normal);

        // A kill-plane reset or a restart moves the ball further in one tick
        // than any speed allows. Treating that as a fall would put a bang on
        // every respawn, which is both wrong and the exact opposite of what a
        // player wants to hear after losing their progress.
        let teleported = audio
            .prev_position
            .is_some_and(|prev| (position - prev).length_squared() > TELEPORT_DISTANCE_SQ);

        if grounded && !audio.was_grounded && !teleported {
            let impact = (-audio.prev_velocity.dot(normal)).max(0.0);
            if impact > LANDING_SPEED {
                let loudness = ((impact - LANDING_SPEED)
                    / (LANDING_FULL_SPEED - LANDING_SPEED))
                    .clamp(0.0, 1.0);
                out.play_at(
                    THUD,
                    state.resets as u64,
                    LANDING_GAIN * loudness,
                    position,
                    &listener,
                    ROLLOFF,
                );
            }
        }

        audio.was_grounded = grounded;
        audio.prev_velocity = velocity.0;
        audio.prev_position = Some(position);
    }

    // -- 4. the win ---------------------------------------------------------
    if state.phase == Phase::Won && audio.fanfare_start.is_none() {
        audio.fanfare_start = Some(tick.0);
        audio.fanfare_played = 0;
    }
    if state.phase == Phase::Playing {
        // A restart re-arms it.
        audio.fanfare_start = None;
        audio.fanfare_played = 0;
    }
    if let Some(start) = audio.fanfare_start {
        let due = audio.fanfare_played;
        if due < FANFARE.len() && tick.0 >= start + due as u64 * FANFARE_SPACING {
            // Play, then immediately re-pitch. Both land in the same tick's
            // batch and the mixer applies a batch in order, so the note is
            // never heard at the preset's root first.
            let voice = out.play(CHIME, due as u64, 0.7, 0.0);
            out.set_param(voice, ParamId::PITCH, FANFARE[due]);
            audio.fanfare_played += 1;
        }
    }
}

/// Has the player done anything at all yet?
///
/// Any key, any button, any touch — the same set of gestures a browser accepts
/// as consent to start audio, expressed in the engine's own input vocabulary so
/// that neither the game nor the engine has to mention a browser.
fn player_is_here(input: &Input) -> bool {
    input.any_held()
        || input.just_pressed_keys().next().is_some()
        || input.just_released_keys().next().is_some()
        || (0..runt_core::input::MOUSE_BUTTONS as u8).any(|b| input.button_just_pressed(b))
        || input.drive() != glam::Vec2::ZERO
}

/// Distance², m², beyond which a one-tick move is a teleport rather than motion.
///
/// `BALL_MAX_SPEED` is 25 m/s and a tick is 1/60 s, so the fastest legal step is
/// ~0.42 m. Four metres is an order of magnitude clear of that and two orders
/// short of a kill-plane respawn.
const TELEPORT_DISTANCE_SQ: f32 = 16.0;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_patch_the_game_plays_is_in_the_bank() {
        // The two `PatchId` types live in crates that do not depend on each
        // other, so this is the join: the names the sim plays must be the names
        // the synth was given.
        let bank = bank();
        for (id, name) in [
            (PICKUP, "pickup"),
            (THUD, "thud"),
            (CHIME, "chime"),
            (AMBIENCE, "ambience"),
        ] {
            assert!(bank.get_by_name(name).is_some(), "{name} missing from bank");
            assert_eq!(
                id.0,
                runt_audio::PatchId::new(name).0,
                "{name} hashes differently on the two sides"
            );
        }
        assert_eq!(bank.len(), 4, "the bank has exactly what the game plays");
    }

    #[test]
    fn the_bank_is_small_enough_to_ship_in_a_processor_option() {
        let bytes = bank().to_bytes().expect("encode");
        assert!(bytes.len() < 512, "{} bytes", bytes.len());
    }
}
