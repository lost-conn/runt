//! Remappable actions (DESIGN §4) — the layer between [`Input`] and gameplay.
//!
//! The engine owns the *mechanism*: a binding table and a per-tick resolution
//! pass. Each game owns its *vocabulary*: an enum of the things that game can be
//! asked to do, implementing [`ActionId`]. The engine has no opinion about
//! whether a game has a "jump" — only about what it means for an action, any
//! action, to be held this tick.
//!
//! # Why this is a separate layer
//!
//! Gameplay that reads `input.held(Key::Space)` has hard-coded a keyboard, a
//! layout and a player's preferences into the sim. Gameplay that reads
//! `actions.held(Jump)` has hard-coded nothing: the mapping from Space (or
//! [`PadButton::South`], or a trigger past 60%) to `Jump` lives in
//! [`Bindings`], which is plain serializable data. The remap story is therefore
//! "edit the data" — a RON file today, a settings screen later, with no change
//! to a single gameplay system in between.
//!
//! # Why it stays honest under replay
//!
//! [`Actions::resolve`] is a pure function of ([`Input`], [`Bindings`]): it
//! reads no clock, allocates nothing, and keeps no state but the previous tick's
//! held bits. A trace records raw [`InputEvent`](crate::InputEvent)s (DESIGN §4)
//! and the actions are **re-derived** from them on playback — so a replay
//! recorded with one binding set and played back with another is not a corrupt
//! trace, it is a different (and reproducible) run. Nothing about which key was
//! pressed ever needs to be stored.

use std::marker::PhantomData;

use bevy_ecs::prelude::{Res, ResMut};
use bevy_ecs::resource::Resource;
use glam::Vec2;
use serde::{Deserialize, Serialize};

use crate::input::{Input, Key, PadButton, PadStick, PadTrigger};

/// Godot's default `deadzone` for an input action, and ours (see
/// [`Bindings::deadzone`]). Kept as a named constant because "0.2" appearing in
/// three files is how two of them end up at 0.15.
pub const DEFAULT_DEADZONE: f32 = 0.2;

/// The widest action set [`Actions`] can hold — one `u32` of bits.
///
/// Not a limit anyone is close to (a whole game's verbs is a dozen), and the
/// alternative is a heap allocation per action set to buy headroom nobody
/// spends. [`Actions::new`] asserts it at compile time, per concrete `A`.
pub const MAX_ACTIONS: usize = 32;

// ---------------------------------------------------------------------------
// The game's vocabulary
// ---------------------------------------------------------------------------

/// A game's action enum, seen from the engine.
///
/// Implemented on a plain fieldless enum, exactly like [`Key`]'s own dense-index
/// pattern:
///
/// ```
/// use runt_core::action::ActionId;
///
/// #[derive(Clone, Copy, Debug, PartialEq)]
/// #[repr(u8)]
/// pub enum Action { Jump, Fire }
///
/// impl ActionId for Action {
///     const COUNT: usize = Action::Fire as usize + 1;
///     fn index(self) -> usize { self as usize }
/// }
/// ```
///
/// `index` must be dense (`0..COUNT`) and stable for the life of the build: it
/// is both the bit position in [`Actions`] and the row in
/// [`Bindings::actions`]. Appending a variant is free; reordering one silently
/// rebinds every action after it, which is what [`Bindings::actions`]'s stored
/// names exist to make detectable.
pub trait ActionId: Copy + Send + Sync + 'static {
    /// Number of distinct actions — the width of the action set.
    const COUNT: usize;

    /// Dense index in `0..COUNT`.
    fn index(self) -> usize;
}

// ---------------------------------------------------------------------------
// Bindings
// ---------------------------------------------------------------------------

/// A direction on an analog stick, for the "treat a stick like a d-pad" case.
///
/// `Up`/`Down` are along `+y`/`-y` and `Right`/`Left` along `+x`/`-x`, matching
/// [`Input::stick`]'s convention (`x` right, `y` up/forward).
#[derive(Clone, Copy, PartialEq, Debug, Serialize, Deserialize)]
pub enum StickDir {
    Up,
    Down,
    Left,
    Right,
}

impl StickDir {
    /// The stick's deflection along this direction — negative when the stick is
    /// pushed the other way, so a single `>` against a positive threshold is the
    /// whole test.
    #[inline]
    fn component(self, v: Vec2) -> f32 {
        match self {
            StickDir::Up => v.y,
            StickDir::Down => -v.y,
            StickDir::Right => v.x,
            StickDir::Left => -v.x,
        }
    }
}

/// One thing that can make an action held.
///
/// Every variant answers a yes/no question about [`Input`] *this tick*; the
/// analog ones do it by comparing a level against a threshold, which is what
/// lets a trigger or a stick stand in for a button (Godot's
/// `InputEventJoypadMotion` with an `axis_value`, in effect).
///
/// The threshold is carried per binding rather than taken from
/// [`Bindings::deadzone`]: the deadzone answers "has the player let go?", while
/// this answers "did the player mean it?", and a menu confirm bound to a
/// half-pulled trigger wants a different number from a walk cycle.
///
/// Serialized as an externally-tagged enum — RON writes `Key(Space)`,
/// `PadTriggerButton(trigger: R2, threshold: 0.6)` — with no `flatten` or
/// `untagged` anywhere, so the file stays hand-editable and the errors stay
/// legible.
#[derive(Clone, Copy, PartialEq, Debug, Serialize, Deserialize)]
pub enum Source {
    Key(Key),
    /// `0` = left, `1` = right, `2` = middle; see [`Input::button_held`].
    MouseButton(u8),
    PadButton(PadButton),
    /// An analog trigger held past `threshold` — analog as button.
    PadTriggerButton { trigger: PadTrigger, threshold: f32 },
    /// An analog stick pushed past `threshold` along `dir` — a stick as a
    /// d-pad, which is what menu navigation will want.
    ///
    /// Compares the **raw** stick, not the deadzone-remapped one: `threshold`
    /// *is* this binding's deadzone, and running it through
    /// [`Bindings::deadzone`] first would mean two knobs fighting over one edge.
    PadStickButton {
        stick: PadStick,
        dir: StickDir,
        threshold: f32,
    },
}

/// The whole remap surface: what makes each action fire, and how movement and
/// look are steered.
///
/// Plain data by design — clone it, ship it in a RON file next to the scene,
/// hand it to a settings screen. It is a [`Resource`] so
/// [`resolve_actions`] can read it, and nothing in the engine ever writes it.
///
/// # Why movement is not just more actions
///
/// A binding table of four "move" actions would give a stick nothing but four
/// booleans, throwing away exactly the analog information a stick exists to
/// supply. So the two analog intents get their own fields: the discrete verbs
/// live in `actions`, the continuous ones in `move_*`/`look_*`.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Resource)]
pub struct Bindings {
    /// One row per action, **index-aligned with [`ActionId::index`]**.
    ///
    /// The `String` is not used for lookup — resolution is by index, which is
    /// what keeps it allocation-free and hash-free (DESIGN §3). It is there so
    /// the serialized file reads as `("jump", [Key(Space)])` rather than as an
    /// anonymous list, and so a future loader can *match by name* when a game
    /// has reordered its enum since the file was written.
    pub actions: Vec<(String, Vec<Source>)>,
    /// Keys that steer movement: `[+x (right), -x (left), +y (forward), -y
    /// (back)]`. Several keys per direction, so WASD and the arrows can both be
    /// live without either being "the" binding.
    pub move_keys: [Vec<Key>; 4],
    /// The stick that steers movement, summed with the keys and with
    /// [`Input::drive`].
    pub move_stick: Option<PadStick>,
    /// The stick that steers looking.
    pub look_stick: Option<PadStick>,
    /// Radial deadzone for both sticks, `0..1`. Godot's default is 0.2 and so is
    /// [`DEFAULT_DEADZONE`].
    ///
    /// **Radial, not per-axis**: the test is on the vector's length, so a stick
    /// pushed diagonally at 0.15 per axis is still centred rather than
    /// producing a phantom diagonal.
    ///
    /// One game-wide value, where Godot stores a deadzone per action. A
    /// deliberate simplification: the number describes the *hardware's* slop, so
    /// per-action values are a way of spelling "this stick is worn out" twelve
    /// times. If a game ever needs a per-action threshold it already has one —
    /// [`Source::PadStickButton`]'s.
    pub deadzone: f32,
    /// Multiplier a *camera* is expected to apply to [`Actions::look_vector`].
    /// Stored here because it belongs to the same settings blob; deliberately
    /// **not** applied by [`Actions::resolve`] (see [`Actions::look_vector`]).
    pub look_sensitivity: f32,
}

impl Bindings {
    /// The sources bound to action `index`, or an empty slice if the table is
    /// shorter than the game's action enum.
    ///
    /// The fallback is the point: a binding file written before an action
    /// existed must load and leave that action simply unbound, not fail. An
    /// unbound action is never held, which is the correct behaviour for a verb
    /// the player has not been given a way to perform.
    #[inline]
    pub fn sources(&self, index: usize) -> &[Source] {
        match self.actions.get(index) {
            Some((_, sources)) => sources,
            None => &[],
        }
    }
}

impl Default for Bindings {
    /// No actions (the engine has no vocabulary to guess at), WASD movement,
    /// left stick to move, right stick to look, Godot's deadzone.
    ///
    /// The movement half can be defaulted because it is engine vocabulary —
    /// there are exactly four directions and every game means the same thing by
    /// them. The action half cannot, so it starts empty and every action is
    /// unbound until a game fills it in.
    fn default() -> Bindings {
        Bindings {
            actions: Vec::new(),
            move_keys: [
                vec![Key::D, Key::Right],
                vec![Key::A, Key::Left],
                vec![Key::W, Key::Up],
                vec![Key::S, Key::Down],
            ],
            move_stick: Some(PadStick::Left),
            look_stick: Some(PadStick::Right),
            deadzone: DEFAULT_DEADZONE,
            look_sensitivity: 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Resolved actions
// ---------------------------------------------------------------------------

/// One tick's resolved action state for the vocabulary `A`.
///
/// A [`Resource`] per action enum, filled by [`resolve_actions`] at the head of
/// `FixedSim` and read by every gameplay system after it. Four `u32`s and two
/// vectors — small enough to be `Copy`-cheap to pass around, and free of any
/// allocation, so a tick's action work is a handful of bit operations.
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct Actions<A: ActionId> {
    held: u32,
    /// Last tick's `held`, and the *only* state carried between ticks. The
    /// edges are derived from it rather than stored across the boundary.
    prev: u32,
    pressed: u32,
    released: u32,
    move_vec: Vec2,
    look_vec: Vec2,
    _marker: PhantomData<A>,
}

impl<A: ActionId> Actions<A> {
    pub fn new() -> Actions<A> {
        // Per concrete `A`, at monomorphization: a vocabulary too wide for the
        // bitset fails the build rather than silently shifting off the end.
        const { assert!(A::COUNT <= MAX_ACTIONS, "an ActionId may have at most 32 actions") };
        Actions {
            held: 0,
            prev: 0,
            pressed: 0,
            released: 0,
            move_vec: Vec2::ZERO,
            look_vec: Vec2::ZERO,
            _marker: PhantomData,
        }
    }

    /// Recompute every action and both analog vectors from this tick's input.
    ///
    /// Pure: same ([`Input`], [`Bindings`]) plus the same previous state gives
    /// the same result, always, with no allocation (the source lists are walked
    /// as slices). That is what lets a replay re-derive actions from a raw event
    /// trace — see the module docs.
    ///
    /// # Why the edges are computed here
    ///
    /// The obvious implementation forwards [`Input::just_pressed`] and friends:
    /// an action is "just pressed" if any of its sources was. It is wrong twice
    /// over. A trigger crossing its threshold has no edge in [`Input`] at all —
    /// it is a level — so an analog binding would never fire. And two sources
    /// bound to one action would produce *two* presses if both went down in one
    /// tick, and a spurious release when the first of two held sources let go.
    ///
    /// Diffing this tick's held bits against last tick's fixes both: the edge
    /// belongs to the *action*, not to any particular source, so a trigger
    /// crossing 0.6 is indistinguishable from a key going down, and an action
    /// held by two things is held once.
    pub fn resolve(&mut self, input: &Input, bindings: &Bindings) {
        let mut held = 0u32;
        for index in 0..A::COUNT {
            let active = bindings
                .sources(index)
                .iter()
                .any(|source| source_active(source, input));
            if active {
                held |= 1u32 << index;
            }
        }

        self.pressed = held & !self.prev;
        self.released = !held & self.prev;
        self.held = held;
        self.prev = held;

        let deadzone = sane_deadzone(bindings.deadzone);
        let stick = match bindings.move_stick {
            Some(stick) => deflection(input.stick(stick), deadzone),
            None => Vec2::ZERO,
        };
        // Three contributors summed, then one clamp: a player using the
        // keyboard and a stick at once gets the union of their intent, not
        // double the speed.
        self.move_vec = clamp_to_unit(key_vector(input, bindings) + input.drive() + stick);

        self.look_vec = match bindings.look_stick {
            Some(stick) => deflection(input.stick(stick), deadzone),
            None => Vec2::ZERO,
        };
    }

    #[inline]
    pub fn held(&self, action: A) -> bool {
        self.held & bit(action) != 0
    }

    /// Whether `action` became held *this tick* — see [`resolve`](Actions::resolve)
    /// for why this is not simply forwarded from [`Input`].
    #[inline]
    pub fn just_pressed(&self, action: A) -> bool {
        self.pressed & bit(action) != 0
    }

    #[inline]
    pub fn just_released(&self, action: A) -> bool {
        self.released & bit(action) != 0
    }

    /// Movement intent: `x` right, `y` forward, magnitude `0..=1`.
    ///
    /// Keys, the touch drive stick ([`Input::drive`]) and the bound pad stick,
    /// summed and clamped to the unit disc — so a diagonal on the keyboard is a
    /// unit vector, not the √2 that `(1, 1)` would otherwise be.
    #[inline]
    pub fn move_vector(&self) -> Vec2 {
        self.move_vec
    }

    /// Look intent: the bound stick's deflection past the deadzone, `0..=1`,
    /// with **no** sensitivity applied.
    ///
    /// [`Bindings::look_sensitivity`] is deliberately left to the consumer.
    /// Sensitivity is a camera concern — how many radians a full deflection is
    /// worth depends on the camera, the tick length and whether the game is
    /// inverting the axis — while this function's job is to report how far the
    /// player pushed the stick. Baking a multiplier in here would make a value
    /// that is nominally `0..=1` silently not be, and would put the camera's
    /// tuning inside the layer replays re-derive.
    #[inline]
    pub fn look_vector(&self) -> Vec2 {
        self.look_vec
    }
}

impl<A: ActionId> Default for Actions<A> {
    fn default() -> Actions<A> {
        Actions::new()
    }
}

/// The bit `action` occupies. Debug-asserted rather than masked: an out-of-range
/// index means [`ActionId::index`] disagrees with [`ActionId::COUNT`], which is
/// a bug in the game's enum and not something to paper over at runtime.
#[inline]
fn bit<A: ActionId>(action: A) -> u32 {
    let index = action.index();
    debug_assert!(index < A::COUNT, "ActionId::index out of range");
    1u32 << (index % MAX_ACTIONS)
}

/// Is this source active *right now*? The one place a [`Source`] meets
/// [`Input`].
#[inline]
fn source_active(source: &Source, input: &Input) -> bool {
    match *source {
        Source::Key(key) => input.held(key),
        Source::MouseButton(button) => input.button_held(button),
        Source::PadButton(button) => input.pad_held(button),
        Source::PadTriggerButton { trigger, threshold } => input.trigger(trigger) > threshold,
        Source::PadStickButton {
            stick,
            dir,
            threshold,
        } => dir.component(input.stick(stick)) > threshold,
    }
}

/// The keyboard's contribution to movement: `±1` per axis, from whether *any*
/// key bound to that direction is held.
#[inline]
fn key_vector(input: &Input, bindings: &Bindings) -> Vec2 {
    let axis = |list: &Vec<Key>| list.iter().any(|key| input.held(*key));
    let right = axis(&bindings.move_keys[0]) as i32 as f32;
    let left = axis(&bindings.move_keys[1]) as i32 as f32;
    let forward = axis(&bindings.move_keys[2]) as i32 as f32;
    let back = axis(&bindings.move_keys[3]) as i32 as f32;
    // Opposite keys held together cancel, which is the same thing every engine
    // does and the only answer that does not depend on press order.
    Vec2::new(right - left, forward - back)
}

/// A stick's deflection past the deadzone, remapped so the usable range starts
/// at zero again.
///
/// `len <= dz` is centred; above it the magnitude is `(len - dz) / (1 - dz)`,
/// along the stick's own direction. This is Godot's `Input.get_vector` and
/// runt's own `VirtualStick::deflection`, and the property that matters is
/// **starting from zero**: the naive version (zero inside the deadzone, the raw
/// length outside) jumps from 0 to 0.2 the instant the stick leaves the dead
/// region, so a character cannot be nudged. Here 0.2001 produces ~0.0005 and the
/// range is continuous.
#[inline]
fn deflection(raw: Vec2, deadzone: f32) -> Vec2 {
    let len = raw.length();
    if !len.is_finite() || len <= deadzone {
        return Vec2::ZERO;
    }
    let scale = ((len - deadzone) / (1.0 - deadzone)).min(1.0);
    // `len > deadzone >= 0` here, so the division is safe.
    raw * (scale / len)
}

/// Clamp a deadzone a binding file might have got wrong. A non-finite value
/// becomes zero; `1.0` and above centres every stick, which falls out of
/// [`deflection`]'s `len <= deadzone` test for any legal stick.
#[inline]
fn sane_deadzone(deadzone: f32) -> f32 {
    if deadzone.is_finite() {
        deadzone.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Pull a vector back onto the unit disc, leaving anything inside it alone.
#[inline]
fn clamp_to_unit(v: Vec2) -> Vec2 {
    if !v.is_finite() {
        Vec2::ZERO
    } else if v.length_squared() > 1.0 {
        v.normalize()
    } else {
        v
    }
}

// ---------------------------------------------------------------------------
// System
// ---------------------------------------------------------------------------

/// `FixedSim` (head): turn this tick's [`Input`] into [`Actions<A>`].
///
/// The ready-made system a game registers **first** in its `FixedSim` schedule,
/// before anything that reads actions — the ordering is explicit, as DESIGN §3
/// requires, and getting it wrong means gameplay reads last tick's verbs.
///
/// Generic over the game's vocabulary, so a game writes
/// `schedule.add_systems(resolve_actions::<Action>)` and is done.
pub fn resolve_actions<A: ActionId>(
    mut actions: ResMut<Actions<A>>,
    input: Res<Input>,
    bindings: Res<Bindings>,
) {
    actions.resolve(&input, &bindings);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::InputEvent;

    #[derive(Clone, Copy, Debug, PartialEq)]
    #[repr(u8)]
    enum Act {
        Jump,
        Fire,
    }

    impl ActionId for Act {
        const COUNT: usize = Act::Fire as usize + 1;
        fn index(self) -> usize {
            self as usize
        }
    }

    /// Bindings with `Jump` and `Fire` bound to whatever the test needs.
    fn bindings(jump: Vec<Source>, fire: Vec<Source>) -> Bindings {
        Bindings {
            actions: vec![("jump".into(), jump), ("fire".into(), fire)],
            ..Bindings::default()
        }
    }

    fn tick(
        actions: &mut Actions<Act>,
        input: &mut Input,
        bindings: &Bindings,
        events: impl IntoIterator<Item = InputEvent>,
    ) {
        input.begin_tick(events);
        actions.resolve(input, bindings);
    }

    #[test]
    fn any_bound_source_makes_the_action_held() {
        let b = bindings(
            vec![
                Source::Key(Key::Space),
                Source::PadButton(PadButton::South),
                Source::MouseButton(0),
            ],
            vec![],
        );
        let mut input = Input::new();
        let mut actions = Actions::<Act>::new();

        tick(&mut actions, &mut input, &b, [InputEvent::KeyDown(Key::Space)]);
        assert!(actions.held(Act::Jump));

        tick(&mut actions, &mut input, &b, [InputEvent::KeyUp(Key::Space)]);
        assert!(!actions.held(Act::Jump));

        tick(
            &mut actions,
            &mut input,
            &b,
            [InputEvent::PadButton {
                button: PadButton::South,
                pressed: true,
            }],
        );
        assert!(actions.held(Act::Jump), "the pad is as good as the key");

        tick(
            &mut actions,
            &mut input,
            &b,
            [
                InputEvent::PadButton {
                    button: PadButton::South,
                    pressed: false,
                },
                InputEvent::MouseButton {
                    button: 0,
                    pressed: true,
                },
            ],
        );
        assert!(actions.held(Act::Jump), "and so is the mouse");
    }

    #[test]
    fn an_unbound_action_is_never_held() {
        // A binding file shorter than the game's enum must load and leave the
        // tail unbound rather than fail.
        let b = Bindings {
            actions: vec![("jump".into(), vec![Source::Key(Key::Space)])],
            ..Bindings::default()
        };
        assert!(b.sources(1).is_empty());

        let mut input = Input::new();
        let mut actions = Actions::<Act>::new();
        tick(&mut actions, &mut input, &b, [InputEvent::KeyDown(Key::Space)]);
        assert!(actions.held(Act::Jump));
        assert!(!actions.held(Act::Fire));
        assert!(!actions.just_pressed(Act::Fire));
    }

    #[test]
    fn a_trigger_crossing_its_threshold_presses_exactly_once() {
        let b = bindings(
            vec![],
            vec![Source::PadTriggerButton {
                trigger: PadTrigger::R2,
                threshold: 0.5,
            }],
        );
        let mut input = Input::new();
        let mut actions = Actions::<Act>::new();

        // Below the threshold: nothing, however much the level moves.
        tick(
            &mut actions,
            &mut input,
            &b,
            [InputEvent::PadTrigger {
                trigger: PadTrigger::R2,
                value: 0.4,
            }],
        );
        assert!(!actions.held(Act::Fire));

        tick(
            &mut actions,
            &mut input,
            &b,
            [InputEvent::PadTrigger {
                trigger: PadTrigger::R2,
                value: 0.6,
            }],
        );
        assert!(actions.held(Act::Fire));
        assert!(
            actions.just_pressed(Act::Fire),
            "crossing the threshold is a press, exactly like a key going down"
        );

        // Pulled harder, still one press. The trigger is a level and reports no
        // edge of its own; the edge belongs to the action.
        tick(
            &mut actions,
            &mut input,
            &b,
            [InputEvent::PadTrigger {
                trigger: PadTrigger::R2,
                value: 1.0,
            }],
        );
        assert!(actions.held(Act::Fire));
        assert!(!actions.just_pressed(Act::Fire));

        tick(
            &mut actions,
            &mut input,
            &b,
            [InputEvent::PadTrigger {
                trigger: PadTrigger::R2,
                value: 0.1,
            }],
        );
        assert!(!actions.held(Act::Fire));
        assert!(actions.just_released(Act::Fire));
    }

    #[test]
    fn two_sources_held_produce_one_press_and_one_release() {
        let b = bindings(
            vec![Source::Key(Key::Space), Source::PadButton(PadButton::South)],
            vec![],
        );
        let mut input = Input::new();
        let mut actions = Actions::<Act>::new();

        // Both go down in the same tick: one press, not two (a bitfield cannot
        // fire twice, which is exactly the point of resolving at action level).
        tick(
            &mut actions,
            &mut input,
            &b,
            [
                InputEvent::KeyDown(Key::Space),
                InputEvent::PadButton {
                    button: PadButton::South,
                    pressed: true,
                },
            ],
        );
        assert!(actions.held(Act::Jump));
        assert!(actions.just_pressed(Act::Jump));

        // One lets go: still held by the other, and *no* release edge.
        tick(&mut actions, &mut input, &b, [InputEvent::KeyUp(Key::Space)]);
        assert!(actions.held(Act::Jump));
        assert!(
            !actions.just_released(Act::Jump),
            "a source releasing is not the action releasing"
        );
        assert!(!actions.just_pressed(Act::Jump));

        tick(
            &mut actions,
            &mut input,
            &b,
            [InputEvent::PadButton {
                button: PadButton::South,
                pressed: false,
            }],
        );
        assert!(!actions.held(Act::Jump));
        assert!(actions.just_released(Act::Jump));

        // And the release is not re-announced on the tick after.
        tick(&mut actions, &mut input, &b, []);
        assert!(!actions.just_released(Act::Jump));
    }

    #[test]
    fn movement_keys_cancel_and_diagonals_are_clamped_to_unit_length() {
        let b = bindings(vec![], vec![]);
        let mut input = Input::new();
        let mut actions = Actions::<Act>::new();

        tick(&mut actions, &mut input, &b, [InputEvent::KeyDown(Key::W)]);
        assert_eq!(actions.move_vector(), Vec2::new(0.0, 1.0));

        // W+D is raw (1, 1) — length √2 — and must come out at 1.
        tick(&mut actions, &mut input, &b, [InputEvent::KeyDown(Key::D)]);
        let v = actions.move_vector();
        assert!((v.length() - 1.0).abs() < 1e-6, "{v:?}");
        assert!((v.x - v.y).abs() < 1e-6, "still a diagonal: {v:?}");

        // The alternate binding for the same direction is not a second push.
        tick(&mut actions, &mut input, &b, [InputEvent::KeyDown(Key::Up)]);
        assert!((actions.move_vector().length() - 1.0).abs() < 1e-6);

        // Opposites cancel.
        tick(
            &mut actions,
            &mut input,
            &b,
            [InputEvent::KeyDown(Key::A), InputEvent::KeyDown(Key::S)],
        );
        assert_eq!(actions.move_vector(), Vec2::ZERO);
    }

    #[test]
    fn the_radial_deadzone_remaps_deflection_from_zero() {
        let b = bindings(vec![], vec![]);
        let mut input = Input::new();
        let mut actions = Actions::<Act>::new();
        assert_eq!(b.deadzone, DEFAULT_DEADZONE);

        // Exactly at the deadzone: centred.
        tick(
            &mut actions,
            &mut input,
            &b,
            [InputEvent::PadStick {
                stick: PadStick::Left,
                dir: Vec2::new(0.0, 0.2),
            }],
        );
        assert_eq!(actions.move_vector(), Vec2::ZERO);

        // 0.6 → (0.6 - 0.2) / 0.8 = 0.5, along the same direction.
        tick(
            &mut actions,
            &mut input,
            &b,
            [InputEvent::PadStick {
                stick: PadStick::Left,
                dir: Vec2::new(0.6, 0.0),
            }],
        );
        let v = actions.move_vector();
        assert!((v.x - 0.5).abs() < 1e-5, "{v:?}");
        assert!(v.y.abs() < 1e-6);

        // Full deflection is still full deflection.
        tick(
            &mut actions,
            &mut input,
            &b,
            [InputEvent::PadStick {
                stick: PadStick::Left,
                dir: Vec2::new(0.0, -1.0),
            }],
        );
        assert!((actions.move_vector() - Vec2::new(0.0, -1.0)).length() < 1e-6);

        // And the range is continuous at the edge: a hair past the deadzone is a
        // hair of movement, not a 0.2 jump.
        tick(
            &mut actions,
            &mut input,
            &b,
            [InputEvent::PadStick {
                stick: PadStick::Left,
                dir: Vec2::new(0.2001, 0.0),
            }],
        );
        let v = actions.move_vector();
        assert!(v.x > 0.0 && v.x < 0.001, "{v:?}");
    }

    #[test]
    fn a_diagonal_inside_the_deadzone_is_centred_not_a_phantom_diagonal() {
        // Per-axis deadzones let (0.15, 0.15) — length 0.21 — through as a
        // perfect diagonal. A radial one does not.
        let b = bindings(vec![], vec![]);
        let mut input = Input::new();
        let mut actions = Actions::<Act>::new();
        tick(
            &mut actions,
            &mut input,
            &b,
            [InputEvent::PadStick {
                stick: PadStick::Left,
                dir: Vec2::new(0.13, 0.13),
            }],
        );
        assert_eq!(actions.move_vector(), Vec2::ZERO);
    }

    #[test]
    fn keys_drive_and_stick_sum_then_clamp() {
        let b = bindings(vec![], vec![]);
        let mut input = Input::new();
        let mut actions = Actions::<Act>::new();

        // All three pushing forward at once: the union of intent, capped at 1.
        tick(
            &mut actions,
            &mut input,
            &b,
            [
                InputEvent::KeyDown(Key::W),
                InputEvent::TouchDrive {
                    dir: Vec2::new(0.0, 1.0),
                },
                InputEvent::PadStick {
                    stick: PadStick::Left,
                    dir: Vec2::new(0.0, 1.0),
                },
            ],
        );
        let v = actions.move_vector();
        assert!((v - Vec2::new(0.0, 1.0)).length() < 1e-6, "{v:?}");

        // Keys forward, stick right: a genuine diagonal, still unit length.
        tick(
            &mut actions,
            &mut input,
            &b,
            [
                InputEvent::TouchDrive { dir: Vec2::ZERO },
                InputEvent::PadStick {
                    stick: PadStick::Left,
                    dir: Vec2::new(1.0, 0.0),
                },
            ],
        );
        let v = actions.move_vector();
        assert!((v.length() - 1.0).abs() < 1e-6, "{v:?}");
        assert!(v.x > 0.0 && v.y > 0.0, "{v:?}");
    }

    #[test]
    fn the_drive_stick_alone_moves_without_any_binding() {
        // `Input::drive` is not bound to anything — a touch host's virtual stick
        // always steers movement.
        let b = Bindings {
            move_keys: [vec![], vec![], vec![], vec![]],
            move_stick: None,
            ..bindings(vec![], vec![])
        };
        let mut input = Input::new();
        let mut actions = Actions::<Act>::new();
        tick(
            &mut actions,
            &mut input,
            &b,
            [InputEvent::TouchDrive {
                dir: Vec2::new(0.5, 0.0),
            }],
        );
        assert_eq!(actions.move_vector(), Vec2::new(0.5, 0.0));
    }

    #[test]
    fn look_reports_raw_deflection_with_no_sensitivity_applied() {
        let b = Bindings {
            look_sensitivity: 10.0,
            ..bindings(vec![], vec![])
        };
        let mut input = Input::new();
        let mut actions = Actions::<Act>::new();
        tick(
            &mut actions,
            &mut input,
            &b,
            [InputEvent::PadStick {
                stick: PadStick::Right,
                dir: Vec2::new(0.6, 0.0),
            }],
        );
        let v = actions.look_vector();
        assert!(
            (v.x - 0.5).abs() < 1e-5,
            "sensitivity is the camera's to apply, not ours: {v:?}"
        );
        assert!(v.length() <= 1.0);
        // And the move stick is a different stick.
        assert_eq!(actions.move_vector(), Vec2::ZERO);
    }

    #[test]
    fn an_unbound_look_stick_reports_nothing() {
        let b = Bindings {
            look_stick: None,
            ..bindings(vec![], vec![])
        };
        let mut input = Input::new();
        let mut actions = Actions::<Act>::new();
        tick(
            &mut actions,
            &mut input,
            &b,
            [InputEvent::PadStick {
                stick: PadStick::Right,
                dir: Vec2::new(1.0, 1.0),
            }],
        );
        assert_eq!(actions.look_vector(), Vec2::ZERO);
    }

    #[test]
    fn stick_directions_test_the_signed_component() {
        let b = bindings(
            vec![Source::PadStickButton {
                stick: PadStick::Left,
                dir: StickDir::Up,
                threshold: 0.5,
            }],
            vec![Source::PadStickButton {
                stick: PadStick::Left,
                dir: StickDir::Left,
                threshold: 0.5,
            }],
        );
        let mut input = Input::new();
        let mut actions = Actions::<Act>::new();

        // Pushed up: Up fires, Left does not.
        tick(
            &mut actions,
            &mut input,
            &b,
            [InputEvent::PadStick {
                stick: PadStick::Left,
                dir: Vec2::new(0.0, 0.9),
            }],
        );
        assert!(actions.held(Act::Jump) && !actions.held(Act::Fire));

        // Pushed *down*: neither. A negative component must not read as a big
        // positive one.
        tick(
            &mut actions,
            &mut input,
            &b,
            [InputEvent::PadStick {
                stick: PadStick::Left,
                dir: Vec2::new(0.0, -0.9),
            }],
        );
        assert!(!actions.held(Act::Jump) && !actions.held(Act::Fire));

        // Pushed left (negative x): the Left binding fires.
        tick(
            &mut actions,
            &mut input,
            &b,
            [InputEvent::PadStick {
                stick: PadStick::Left,
                dir: Vec2::new(-0.9, 0.0),
            }],
        );
        assert!(!actions.held(Act::Jump) && actions.held(Act::Fire));

        // Pushed right: neither, for the same reason as down.
        tick(
            &mut actions,
            &mut input,
            &b,
            [InputEvent::PadStick {
                stick: PadStick::Left,
                dir: Vec2::new(0.9, 0.0),
            }],
        );
        assert!(!actions.held(Act::Fire));
    }

    #[test]
    fn resolve_is_a_pure_function_of_input_and_bindings() {
        let b = bindings(
            vec![Source::Key(Key::Space)],
            vec![Source::PadTriggerButton {
                trigger: PadTrigger::L2,
                threshold: 0.3,
            }],
        );
        let events = [
            InputEvent::KeyDown(Key::Space),
            InputEvent::KeyDown(Key::W),
            InputEvent::PadTrigger {
                trigger: PadTrigger::L2,
                value: 0.7,
            },
            InputEvent::PadStick {
                stick: PadStick::Right,
                dir: Vec2::new(0.4, -0.9),
            },
        ];

        let run = || {
            let mut input = Input::new();
            let mut actions = Actions::<Act>::new();
            input.begin_tick(events);
            actions.resolve(&input, &b);
            actions
        };
        let first = run();
        let second = run();
        assert_eq!(first, second);

        // Resolving the same Input twice into the same Actions is idempotent in
        // `held` and drops the edges — the second call finds `prev == held`.
        let mut input = Input::new();
        let mut actions = Actions::<Act>::new();
        input.begin_tick(events);
        actions.resolve(&input, &b);
        assert!(actions.just_pressed(Act::Jump));
        actions.resolve(&input, &b);
        assert!(actions.held(Act::Jump));
        assert!(!actions.just_pressed(Act::Jump));
    }

    #[test]
    fn focus_loss_releases_every_action() {
        let b = bindings(
            vec![Source::Key(Key::Space)],
            vec![Source::PadTriggerButton {
                trigger: PadTrigger::R2,
                threshold: 0.5,
            }],
        );
        let mut input = Input::new();
        let mut actions = Actions::<Act>::new();
        tick(
            &mut actions,
            &mut input,
            &b,
            [
                InputEvent::KeyDown(Key::Space),
                InputEvent::KeyDown(Key::W),
                InputEvent::PadTrigger {
                    trigger: PadTrigger::R2,
                    value: 1.0,
                },
            ],
        );
        assert!(actions.held(Act::Jump) && actions.held(Act::Fire));

        tick(&mut actions, &mut input, &b, [InputEvent::FocusLost]);
        assert!(!actions.held(Act::Jump) && !actions.held(Act::Fire));
        assert!(actions.just_released(Act::Jump) && actions.just_released(Act::Fire));
        assert_eq!(actions.move_vector(), Vec2::ZERO);
    }

    #[test]
    fn a_tick_sequence_reads_as_a_gameplay_script() {
        // The mini integration: raw events in, verbs out, over several ticks.
        let b = bindings(
            vec![Source::Key(Key::Space), Source::PadButton(PadButton::South)],
            vec![
                Source::MouseButton(0),
                Source::PadTriggerButton {
                    trigger: PadTrigger::R2,
                    threshold: 0.5,
                },
            ],
        );
        let mut input = Input::new();
        let mut actions = Actions::<Act>::new();

        // Tick 1: nothing.
        tick(&mut actions, &mut input, &b, []);
        assert!(!actions.held(Act::Jump) && !actions.held(Act::Fire));
        assert_eq!(actions.move_vector(), Vec2::ZERO);

        // Tick 2: walk forward-right and jump.
        tick(
            &mut actions,
            &mut input,
            &b,
            [
                InputEvent::KeyDown(Key::W),
                InputEvent::KeyDown(Key::D),
                InputEvent::KeyDown(Key::Space),
            ],
        );
        assert!(actions.just_pressed(Act::Jump));
        assert!((actions.move_vector().length() - 1.0).abs() < 1e-6);

        // Tick 3: still walking, jump held (no new press), trigger squeezed.
        tick(
            &mut actions,
            &mut input,
            &b,
            [InputEvent::PadTrigger {
                trigger: PadTrigger::R2,
                value: 0.8,
            }],
        );
        assert!(actions.held(Act::Jump) && !actions.just_pressed(Act::Jump));
        assert!(actions.just_pressed(Act::Fire));

        // Tick 4: let go of the jump key, keep the pad button — jump survives.
        tick(
            &mut actions,
            &mut input,
            &b,
            [
                InputEvent::PadButton {
                    button: PadButton::South,
                    pressed: true,
                },
                InputEvent::KeyUp(Key::Space),
            ],
        );
        assert!(actions.held(Act::Jump) && !actions.just_released(Act::Jump));

        // Tick 5: stop. Everything releases exactly once.
        tick(
            &mut actions,
            &mut input,
            &b,
            [
                InputEvent::KeyUp(Key::W),
                InputEvent::KeyUp(Key::D),
                InputEvent::PadButton {
                    button: PadButton::South,
                    pressed: false,
                },
                InputEvent::PadTrigger {
                    trigger: PadTrigger::R2,
                    value: 0.0,
                },
            ],
        );
        assert!(actions.just_released(Act::Jump) && actions.just_released(Act::Fire));
        assert_eq!(actions.move_vector(), Vec2::ZERO);

        tick(&mut actions, &mut input, &b, []);
        assert!(!actions.just_released(Act::Jump) && !actions.just_released(Act::Fire));
    }

    #[test]
    fn bindings_round_trip_through_ron() {
        // The remap story is "edit the data", so the data has to survive a file.
        let b = bindings(
            vec![
                Source::Key(Key::Space),
                Source::PadButton(PadButton::South),
                Source::MouseButton(1),
                Source::PadTriggerButton {
                    trigger: PadTrigger::R2,
                    threshold: 0.6,
                },
                Source::PadStickButton {
                    stick: PadStick::Left,
                    dir: StickDir::Up,
                    threshold: 0.5,
                },
            ],
            vec![],
        );
        let text = ron::ser::to_string_pretty(&b, ron::ser::PrettyConfig::new())
            .expect("bindings serialize");
        assert!(text.contains("jump"), "names are there to be read: {text}");
        let back: Bindings = ron::from_str(&text).expect("bindings parse");
        assert_eq!(back, b);
    }

    #[test]
    fn a_broken_deadzone_does_not_produce_a_broken_stick() {
        let mut input = Input::new();
        let mut actions = Actions::<Act>::new();

        for deadzone in [-1.0, 0.0, 1.0, 2.0, f32::NAN] {
            let b = Bindings {
                deadzone,
                ..bindings(vec![], vec![])
            };
            tick(
                &mut actions,
                &mut input,
                &b,
                [InputEvent::PadStick {
                    stick: PadStick::Left,
                    dir: Vec2::new(0.7, 0.0),
                }],
            );
            let v = actions.move_vector();
            assert!(v.is_finite() && v.length() <= 1.0 + 1e-6, "{deadzone}: {v:?}");
        }
    }
}
