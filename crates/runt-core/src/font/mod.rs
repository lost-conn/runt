//! Bitmap fonts: metrics, an atlas, and the layout that turns a `&str` into
//! [`UiQuad`]s — with **no typeface in the engine**.
//!
//! This module is the *code* half of text. The pixels are game content, exactly
//! like [`UiAtlasImage`] says they are: a [`BitmapFont`] is a table of glyph
//! rectangles and advances plus the handle of an atlas somebody else baked, and
//! everything here is arithmetic over that table. `runt-core` therefore ships a
//! text renderer and no letters, which is the split that lets two games look
//! nothing alike without either of them forking the layout.
//!
//! ```text
//! font-bake (dev tool)      the game               runt-core
//! ─────────────────────     ────────────────       ───────────────────
//! NotoSans-Regular.ttf  →   assets/ui.font    →    FontAsset::from_bytes
//!                           include_bytes!         └─ UiAtlasImage (pixels)
//!                                                  └─ BitmapFont   (metrics)
//!                                                        │
//!                                              width / text / wrap / shape
//! ```
//!
//! # `scale`, and why a font knows about it
//!
//! Every layout call takes a `scale`, in the units a HUD already thinks in: the
//! nominal cell is [`UNIT`] logical pixels, so `scale: 2.0` means "two-pixel
//! text", the same thing it meant when the cell was an 8 × 8 bitmap.
//!
//! A [`BitmapFont`] knows which `scale` it *is*
//! ([`BitmapFont::design_scale`]), and [`BitmapFont::factor`] is therefore
//! **1.0 exactly** when it is drawn at that scale — which is the whole point of
//! baking one rasterization per size. The UI sampler is `Nearest` on purpose
//! (`ui.rs`: `filterable: false`, so the pass stays valid under
//! `downlevel_webgl2_defaults`), so a font resampled to a scale it was not baked
//! at looks like exactly what it is. Bake the sizes you draw.
//!
//! `design_scale` is stored rather than derived from [`px`](BitmapFont::px)
//! because the two are not the same question. `px` is whatever number the
//! rasterizer wanted — for `ab_glyph` it is the ascender-to-descender span, for
//! another it might be the em — and which of those lands on a 2× HUD is a
//! judgement the baker makes once and writes down.
//!
//! # Premultiplied, and white
//!
//! [`UiAtlasImage`]'s contract: the atlas is premultiplied and the fragment is
//! `texel · color`, so a tintable font must write `rgb = a`. [`FontAsset`]
//! stores single-channel coverage and [`FontAsset::image`] expands it, so the
//! invariant is a property of the type rather than of whoever wrote the baker.
//!
//! [`UiAtlasImage`]: crate::ui::UiAtlasImage

use serde::{Deserialize, Serialize};

use crate::texture::TextureHandle;
use crate::ui::{UiAtlasImage, UiBatch, UiQuad};

/// The optional 8 × 8 fallback typeface (`default-font`). See the module.
#[cfg(feature = "default-font")]
pub mod micro;

/// Logical pixels one unit of `scale` stands for.
///
/// Eight, inherited from the 8 × 8 cell every hand-authored micro-font in this
/// lineage used (see [`micro`]), because that is the number every call site's
/// `scale: 2.0` was already written against.
///
/// It is a *nominal* cell and nothing enforces it: a real typeface's line box is
/// [`BitmapFont::line_height`], measured from the ink it actually has, and a
/// baker is expected to choose a size that lands near `UNIT · scale` rather than
/// on it. `UNIT` exists for the geometry that has **no font in hand** — chiefly
/// [`tweak_panel::row_pitch`](crate::tweak_panel::row_pitch), where the hit test
/// and the draw have to agree on a number before either has seen a glyph.
pub const UNIT: f32 = 8.0;

// ---------------------------------------------------------------------------
// Glyphs
// ---------------------------------------------------------------------------

/// One glyph's cell in the atlas, and how to place it.
///
/// Every field is in **atlas texels** at the font's own [`px`](BitmapFont::px);
/// [`BitmapFont::factor`] converts to logical pixels. The layout follows the
/// usual convention so a baker written against any rasterizer can fill it in:
/// the pen sits on the baseline, `bearing_x` walks right to the ink's left edge
/// and `bearing_y` walks *up* to its top edge.
///
/// A glyph with no ink — a space — carries `width == 0`, and every drawing path
/// skips it rather than pushing an empty quad.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Glyph {
    /// Left edge of the ink in the atlas.
    pub x: u16,
    /// Top edge of the ink in the atlas.
    pub y: u16,
    /// Ink width.
    pub width: u16,
    /// Ink height.
    pub height: u16,
    /// Pen → the ink's left edge. Negative for a glyph that overhangs.
    pub bearing_x: i16,
    /// Baseline → the ink's top edge, positive **up**.
    pub bearing_y: i16,
    /// How far the pen steps after this glyph, in texels.
    ///
    /// `f32` rather than an integer because a proportional face's advance is
    /// fractional and rounding it per glyph is how a long line drifts.
    pub advance: f32,
}

impl Glyph {
    /// Is there anything to draw? A space is not.
    pub fn has_ink(&self) -> bool {
        self.width > 0 && self.height > 0
    }
}

/// One kerning pair, in texels.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Kern {
    pub left: u32,
    pub right: u32,
    /// Added to the pen *between* the two, so it is usually negative.
    pub adjust: f32,
}

// ---------------------------------------------------------------------------
// The font
// ---------------------------------------------------------------------------

/// A font: a glyph table, a codepoint map, and the metrics that place them.
///
/// # The codepoint map
///
/// `codepoints` is sorted and parallel to `glyphs`, so a lookup is a binary
/// search over `u32` and the covered set is whatever the baker was asked for —
/// printable ASCII, Latin-1, a handful of hand-picked arrows, or all three.
/// Nothing here assumes a contiguous range, which is the bug a `c as usize - 32`
/// table has the moment somebody wants `→`.
///
/// # The two width answers
///
/// [`width`](BitmapFont::width) is where the **pen** ends and
/// [`ink_width`](BitmapFont::ink_width) is where the **ink** ends. They differ
/// by the last glyph's right side bearing. Right-aligned layout wants the
/// second; a caller composing a line out of pieces wants the first, and
/// [`text`](BitmapFont::text) returns exactly `x + width(…)` so the two can
/// never disagree.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct BitmapFont {
    /// The atlas these cells index. Not serialized: the handle is the *game's*
    /// to choose (see [`TextureHandle`]'s docs on bit 63), so
    /// [`FontAsset::fonts`] stamps it at load.
    #[serde(skip)]
    pub atlas: TextureHandle,
    /// Atlas dimensions, for the uv divide.
    pub atlas_width: u32,
    pub atlas_height: u32,
    /// The size the rasterizer was asked for, in texels. Informational: what it
    /// *meant* is the rasterizer's business, and [`design_scale`] is what layout
    /// uses.
    ///
    /// [`design_scale`]: BitmapFont::design_scale
    pub px: f32,
    /// The `scale` this font draws texel-for-pixel at.
    pub design_scale: f32,
    /// Baseline → the top of the tallest ink, in texels.
    pub ascent: f32,
    /// Baseline → the bottom of the deepest ink, positive **down**.
    pub descent: f32,
    /// Pen step between two lines, in texels.
    pub line_height: f32,
    /// The pen step for a codepoint the font does not cover.
    ///
    /// A missing character draws nothing and advances by this, so a HUD that
    /// meets an em dash comes out one space short rather than one glyph wrong.
    pub missing_advance: f32,
    /// Sorted by `codepoints`, and the same length as it.
    pub glyphs: Vec<Glyph>,
    /// Sorted ascending. `codepoints[i]` is drawn by `glyphs[i]`.
    pub codepoints: Vec<u32>,
    /// Sorted by `(left, right)`. Empty is the common case and costs nothing.
    pub kerning: Vec<Kern>,
}

impl BitmapFont {
    /// Texels → logical pixels at `scale`.
    ///
    /// `1.0` exactly when `scale == self.design_scale`, which is what a font
    /// baked per size is for.
    pub fn factor(&self, scale: f32) -> f32 {
        if self.design_scale <= 0.0 {
            return 0.0;
        }
        scale / self.design_scale
    }

    /// The glyph index for `c`, or `None`.
    pub fn index(&self, c: char) -> Option<usize> {
        self.codepoints.binary_search(&(c as u32)).ok()
    }

    /// The glyph for `c`, or `None` — which every path here treats as "advance
    /// and draw nothing", never as a wrong glyph.
    pub fn glyph(&self, c: char) -> Option<&Glyph> {
        self.index(c).map(|i| &self.glyphs[i])
    }

    /// Does this font have a cell for `c`?
    pub fn covers(&self, c: char) -> bool {
        self.index(c).is_some()
    }

    /// The kerning adjustment between two codepoints, in texels. `0.0` when the
    /// font carries no pairs, which is the usual case.
    pub fn kern(&self, left: char, right: char) -> f32 {
        if self.kerning.is_empty() {
            return 0.0;
        }
        let key = (left as u32, right as u32);
        self.kerning
            .binary_search_by(|k| (k.left, k.right).cmp(&key))
            .map(|i| self.kerning[i].adjust)
            .unwrap_or(0.0)
    }

    /// The atlas rectangle a glyph occupies, as `[u0, v0, u1, v1]`.
    ///
    /// Cell-exact rather than half-texel-inset: the UI sampler is `Nearest` with
    /// `ClampToEdge` (`crate::ui`), and quads land on integer pixel boundaries at
    /// the scale the font was baked for, so every sample lands in the middle of
    /// its texel and there is nothing to bleed.
    pub fn uv(&self, glyph: &Glyph) -> [f32; 4] {
        let (w, h) = (self.atlas_width as f32, self.atlas_height as f32);
        if w <= 0.0 || h <= 0.0 {
            return [0.0; 4];
        }
        [
            glyph.x as f32 / w,
            glyph.y as f32 / h,
            (glyph.x as f32 + glyph.width as f32) / w,
            (glyph.y as f32 + glyph.height as f32) / h,
        ]
    }

    /// Walk `text`, handing each character the pen position **in texels** it
    /// starts at, and return where the pen ended.
    ///
    /// The single place the pen is stepped. Everything public here is a fold
    /// over this, which is what makes [`width`](BitmapFont::width) and
    /// [`text`](BitmapFont::text) agree by construction rather than by care.
    fn walk(&self, text: &str, mut visit: impl FnMut(f32, char, Option<&Glyph>)) -> f32 {
        let mut pen = 0.0f32;
        let mut prev: Option<char> = None;
        for c in text.chars() {
            if let Some(p) = prev {
                pen += self.kern(p, c);
            }
            let glyph = self.glyph(c);
            visit(pen, c, glyph);
            pen += glyph.map_or(self.missing_advance, |g| g.advance);
            prev = Some(c);
        }
        pen
    }

    /// The pen step for the whole string, in texels.
    pub fn advance(&self, text: &str) -> f32 {
        self.walk(text, |_, _, _| {})
    }

    /// How wide `text` is at `scale`, in logical pixels — to the **pen**,
    /// including the last glyph's right side bearing.
    pub fn width(&self, text: &str, scale: f32) -> f32 {
        self.advance(text) * self.factor(scale)
    }

    /// How wide `text` is up to its last **ink**, which is what centring and
    /// right-alignment want.
    pub fn ink_width(&self, text: &str, scale: f32) -> f32 {
        let mut right = 0.0f32;
        self.walk(text, |pen, _, glyph| {
            if let Some(g) = glyph {
                if g.has_ink() {
                    right = right.max(pen + g.bearing_x as f32 + g.width as f32);
                }
            }
        });
        right * self.factor(scale)
    }

    /// A line's height in logical pixels — the pen step between two lines.
    pub fn line_height(&self, scale: f32) -> f32 {
        self.line_height * self.factor(scale)
    }

    /// Push `text` into `batch` with the **top of its line box** at `(x, y)`, in
    /// straight-alpha `color`.
    ///
    /// Returns the pen position after the last character — exactly
    /// `x + self.width(text, scale)` — so a caller composing a line out of
    /// pieces (a label, then a value in another colour) never has to re-measure.
    ///
    /// The batch is expected to already be sampling
    /// [`self.atlas`](BitmapFont::atlas); this pushes quads and never touches
    /// [`UiBatch::set_texture`], so a HUD that keeps one atlas for everything
    /// stays a single draw call.
    ///
    /// # Fractional pen, integer quads
    ///
    /// A proportional face's advances are fractional, and the UI sampler is
    /// `Nearest` — so a glyph landing on `x = 15.32` samples its texels a third
    /// of a pixel off centre and comes out with one stem a pixel wider than its
    /// neighbour. Each quad's **corner is therefore rounded to a whole pixel**
    /// while the pen keeps its fraction, which is the usual bitmap-font trade:
    /// the letters stay crisp and a long line does not drift, at the cost of a
    /// sub-pixel wobble in the spacing that nothing at these sizes can see.
    ///
    /// The pen is what [`width`](BitmapFont::width) reports and what this
    /// returns, so the two still agree exactly. What rounding *does* cost is
    /// that [`text_right`](BitmapFont::text_right)'s last glyph can land half a
    /// pixel either side of `right`.
    pub fn text(
        &self,
        batch: &mut UiBatch,
        x: f32,
        y: f32,
        text: &str,
        scale: f32,
        color: [f32; 4],
    ) -> f32 {
        let k = self.factor(scale);
        let ascent = self.ascent;
        let end = self.walk(text, |pen, _, glyph| {
            let Some(g) = glyph else { return };
            if !g.has_ink() {
                return;
            }
            let rect = [
                (x + (pen + g.bearing_x as f32) * k).round(),
                (y + (ascent - g.bearing_y as f32) * k).round(),
                g.width as f32 * k,
                g.height as f32 * k,
            ];
            batch.push(UiQuad::textured(rect, self.uv(g), color));
        });
        x + end * k
    }

    /// [`text`](BitmapFont::text), with the string's ink centred on `cx`.
    pub fn text_centered(
        &self,
        batch: &mut UiBatch,
        cx: f32,
        y: f32,
        line: &str,
        scale: f32,
        color: [f32; 4],
    ) -> f32 {
        let x = cx - self.ink_width(line, scale) * 0.5;
        self.text(batch, x, y, line, scale, color)
    }

    /// [`text`](BitmapFont::text), with the string's ink ending at `right`.
    pub fn text_right(
        &self,
        batch: &mut UiBatch,
        right: f32,
        y: f32,
        line: &str,
        scale: f32,
        color: [f32; 4],
    ) -> f32 {
        let x = right - self.ink_width(line, scale);
        self.text(batch, x, y, line, scale, color)
    }

    /// Break `text` into lines whose ink is no wider than `width` logical
    /// pixels at `scale`.
    ///
    /// Whitespace only, greedy, and no hyphenation: this is a HUD, and a
    /// paragraph engine for one sentence is a paragraph engine nobody
    /// maintains. A single word wider than the line is emitted on its own and
    /// allowed to overflow, which is the failure mode that draws something
    /// rather than looping forever.
    pub fn wrap(&self, text: &str, scale: f32, width: f32) -> Vec<String> {
        let mut lines: Vec<String> = Vec::new();
        let mut line = String::new();
        for word in text.split_whitespace() {
            if line.is_empty() {
                line.push_str(word);
                continue;
            }
            let mark = line.len();
            line.push(' ');
            line.push_str(word);
            if self.ink_width(&line, scale) > width {
                line.truncate(mark);
                lines.push(std::mem::take(&mut line));
                line.push_str(word);
            }
        }
        if !line.is_empty() {
            lines.push(line);
        }
        lines
    }

    /// One glyph drawn as a **shape** rather than as a letter: its ink stretched
    /// to `rect`, ignoring the baseline entirely.
    ///
    /// What a health dot and a thumb-button plate are — `●` scaled to whatever
    /// size the screen wants, from the same atlas as the text.
    pub fn shape(&self, batch: &mut UiBatch, rect: [f32; 4], c: char, color: [f32; 4]) {
        let Some(glyph) = self.glyph(c) else { return };
        if !glyph.has_ink() {
            return;
        }
        batch.push(UiQuad::textured(rect, self.uv(glyph), color));
    }
}

// ---------------------------------------------------------------------------
// The baked asset
// ---------------------------------------------------------------------------

/// What `tools/font-bake` writes and a game `include_bytes!`s: one atlas, and
/// one [`BitmapFont`] per size baked into it.
///
/// # One atlas, many sizes
///
/// A font drawn at four scales is four rasterizations, each fitted to its own
/// pixel grid — that is the entire reason not to bake once and scale. They share
/// **one** atlas image because
/// [`TextureRegistry::insert_image`](crate::bake::TextureRegistry::insert_image)
/// is idempotent by handle: one handle is one set of pixels forever, so four
/// images would mean four handles, four resources and four draw calls for a HUD
/// that wants one.
///
/// # Coverage, not RGBA
///
/// `coverage` is one byte per texel. [`image`](FontAsset::image) expands it to
/// the premultiplied `rgb = a` RGBA8 [`UiAtlasImage`] wants, which is a
/// lossless round trip *and* a quarter of the bytes in the shipped wasm. Size
/// discipline is a stated constraint of this engine and a font atlas is the
/// largest single asset a HUD has.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FontAsset {
    pub width: u32,
    pub height: u32,
    /// `width · height` bytes, row-major: 0 is transparent, 255 is solid.
    pub coverage: Vec<u8>,
    /// One per baked size, sorted by [`BitmapFont::design_scale`] ascending.
    pub sizes: Vec<BitmapFont>,
}

/// What went wrong loading a [`FontAsset`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FontError {
    /// The bytes are not a `FontAsset`.
    Decode(String),
    /// They decoded, but describe something that cannot be drawn.
    Malformed(&'static str),
}

impl std::fmt::Display for FontError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FontError::Decode(e) => write!(f, "font asset did not decode: {e}"),
            FontError::Malformed(what) => write!(f, "font asset is malformed: {what}"),
        }
    }
}

impl std::error::Error for FontError {}

impl FontAsset {
    /// Decode a blob written by [`to_bytes`](FontAsset::to_bytes), checking the
    /// invariants a drawing path would otherwise discover as a panic.
    pub fn from_bytes(bytes: &[u8]) -> Result<FontAsset, FontError> {
        let asset: FontAsset =
            postcard::from_bytes(bytes).map_err(|e| FontError::Decode(e.to_string()))?;
        if asset.width == 0 || asset.height == 0 {
            return Err(FontError::Malformed("atlas has no area"));
        }
        let expected = (asset.width as usize) * (asset.height as usize);
        if asset.coverage.len() != expected {
            return Err(FontError::Malformed("coverage is not width × height"));
        }
        if asset.sizes.is_empty() {
            return Err(FontError::Malformed("no sizes"));
        }
        for size in &asset.sizes {
            if size.glyphs.len() != size.codepoints.len() {
                return Err(FontError::Malformed("glyphs and codepoints disagree"));
            }
            if size.codepoints.windows(2).any(|w| w[0] >= w[1]) {
                return Err(FontError::Malformed("codepoints are not sorted"));
            }
            if size.px <= 0.0 || size.design_scale <= 0.0 {
                return Err(FontError::Malformed("size has no scale"));
            }
        }
        Ok(asset)
    }

    /// The compact byte form. Postcard, like every other blob this engine
    /// caches.
    pub fn to_bytes(&self) -> Result<Vec<u8>, postcard::Error> {
        postcard::to_stdvec(self)
    }

    /// The atlas as the resource the engine uploads once: premultiplied RGBA8,
    /// `rgb = a`, white on transparent.
    pub fn image(&self, handle: TextureHandle) -> UiAtlasImage {
        let mut rgba = Vec::with_capacity(self.coverage.len() * 4);
        for &a in &self.coverage {
            rgba.extend_from_slice(&[a, a, a, a]);
        }
        UiAtlasImage {
            handle,
            width: self.width,
            height: self.height,
            rgba,
        }
    }

    /// The fonts, with the atlas handle and dimensions stamped in.
    pub fn fonts(&self, handle: TextureHandle) -> Vec<BitmapFont> {
        self.sizes
            .iter()
            .map(|size| BitmapFont {
                atlas: handle,
                atlas_width: self.width,
                atlas_height: self.height,
                ..size.clone()
            })
            .collect()
    }
}

/// The font in `sizes` closest to `scale`, by design scale.
///
/// Ties go to the larger, because a glyph asked to shrink reads better than one
/// asked to grow under a `Nearest` sampler. Panics on an empty slice, which
/// [`FontAsset::from_bytes`] has already ruled out.
pub fn nearest(sizes: &[BitmapFont], scale: f32) -> &BitmapFont {
    sizes
        .iter()
        .min_by(|a, b| {
            let (da, db) = (
                (a.design_scale - scale).abs(),
                (b.design_scale - scale).abs(),
            );
            da.total_cmp(&db)
                .then_with(|| b.design_scale.total_cmp(&a.design_scale))
        })
        .expect("a font asset always has at least one size")
}

// ---------------------------------------------------------------------------
// The panel seam
// ---------------------------------------------------------------------------

/// Every [`BitmapFont`] is a [`PanelFont`](crate::tweak_panel::PanelFont).
///
/// The debug overlay used to take a trait a game implemented over its own atlas;
/// now that the engine owns the layout there is nothing left for that shim to
/// do, and a game hands the panel the same font it draws its HUD with.
#[cfg(feature = "reflect")]
impl crate::tweak_panel::PanelFont for BitmapFont {
    fn width(&self, text: &str, scale: f32) -> f32 {
        BitmapFont::width(self, text, scale)
    }

    fn text(&self, batch: &mut UiBatch, x: f32, y: f32, text: &str, scale: f32, color: [f32; 4]) {
        BitmapFont::text(self, batch, x, y, text, scale, color);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 3-glyph monospace stand-in: 5 × 7 ink in a 6-texel advance, baked at
    /// [`UNIT`], so `scale` behaves exactly the way the 8 × 8 micro-font's did.
    fn mono() -> BitmapFont {
        let cell = |i: u16| Glyph {
            x: i * 8,
            y: 0,
            width: 5,
            height: 7,
            bearing_x: 0,
            bearing_y: 7,
            advance: 6.0,
        };
        let space = Glyph {
            advance: 6.0,
            ..Glyph::default()
        };
        BitmapFont {
            atlas: TextureHandle(1),
            atlas_width: 32,
            atlas_height: 8,
            px: UNIT,
            design_scale: 1.0,
            ascent: 7.0,
            descent: 1.0,
            line_height: 8.0,
            missing_advance: 6.0,
            glyphs: vec![space, cell(1), cell(2), cell(3)],
            codepoints: vec![' ' as u32, 'A' as u32, 'B' as u32, 'C' as u32],
            kerning: Vec::new(),
        }
    }

    #[test]
    fn a_font_drawn_at_the_size_it_was_baked_for_needs_no_resampling() {
        let font = mono();
        assert_eq!(font.design_scale, 1.0);
        assert_eq!(font.factor(1.0), 1.0);
        assert_eq!(font.factor(2.0), 2.0);

        // …and one baked for 2× draws 1:1 when a call site says `scale: 2.0`,
        // which is the entire point of storing the design scale.
        let big = BitmapFont {
            px: 24.0,
            design_scale: 2.0,
            ..mono()
        };
        assert_eq!(big.factor(2.0), 1.0);
        assert_eq!(big.factor(1.0), 0.5);
    }

    #[test]
    fn width_is_exactly_where_text_leaves_the_pen() {
        // The invariant right-alignment rests on. Not "close": equal.
        let font = mono();
        let mut batch = UiBatch::new();
        for scale in [1.0f32, 2.0, 3.5] {
            for line in ["", "A", "ABC", "A BC", "A—B"] {
                batch.clear();
                let end = font.text(&mut batch, 10.0, 20.0, line, scale, [1.0; 4]);
                assert_eq!(end, 10.0 + font.width(line, scale), "{line:?} @ {scale}");
            }
        }
    }

    #[test]
    fn a_line_of_text_is_one_quad_per_inked_character() {
        let font = mono();
        let mut batch = UiBatch::new();
        let end = font.text(&mut batch, 10.0, 20.0, "AB CA", 2.0, [1.0; 4]);
        // Five characters advance the pen; the space has no ink.
        assert_eq!(batch.len(), 4);
        assert_eq!(end, 10.0 + 5.0 * 6.0 * 2.0);
        assert_eq!(font.width("AB CA", 2.0), 60.0);
        assert_eq!(font.ink_width("AB CA", 2.0), 58.0);

        // The first quad is where it was asked for, at the scaled ink size.
        assert_eq!(batch.quads[0].rect, [10.0, 20.0, 10.0, 14.0]);
        // …and the second is one advance along.
        assert_eq!(batch.quads[1].rect[0], 10.0 + 12.0);

        // Right-aligned text ends where it was told to.
        batch.clear();
        font.text_right(&mut batch, 100.0, 0.0, "AB", 1.0, [1.0; 4]);
        assert_eq!(batch.quads[1].rect[0] + 5.0, 100.0);

        // Centred text is centred on its ink — to the pixel the quad lands on,
        // since a corner is rounded.
        batch.clear();
        font.text_centered(&mut batch, 50.0, 0.0, "AB", 1.0, [1.0; 4]);
        assert_eq!(batch.quads[0].rect[0], (50.0f32 - 11.0 * 0.5).round());
    }

    #[test]
    fn an_uncovered_character_leaves_a_gap_rather_than_a_wrong_glyph() {
        let font = mono();
        let mut batch = UiBatch::new();
        let end = font.text(&mut batch, 0.0, 0.0, "A—B", 1.0, [1.0; 4]);
        assert_eq!(batch.len(), 2, "the em dash drew something");
        assert_eq!(end, 18.0);
        assert_eq!(batch.quads[1].rect[0], 12.0, "the pen skipped");
        assert!(!font.covers('—'));
        assert!(font.covers('A'));
    }

    #[test]
    fn kerning_moves_the_pen_and_the_measurement_together() {
        let mut font = mono();
        font.kerning = vec![Kern {
            left: 'A' as u32,
            right: 'B' as u32,
            adjust: -1.0,
        }];
        assert_eq!(font.kern('A', 'B'), -1.0);
        assert_eq!(font.kern('B', 'A'), 0.0);
        assert_eq!(font.width("AB", 1.0), 11.0);
        let mut batch = UiBatch::new();
        let end = font.text(&mut batch, 0.0, 0.0, "AB", 1.0, [1.0; 4]);
        assert_eq!(end, 11.0);
        assert_eq!(batch.quads[1].rect[0], 5.0);
    }

    #[test]
    fn a_proportional_glyph_is_placed_from_its_bearings() {
        // A comma: narrow ink, low, and offset right of the pen.
        let mut font = mono();
        // `codepoints` is sorted and parallel to `glyphs`, so a comma goes
        // between the space and the letters rather than on the end.
        let at = font.codepoints.partition_point(|cp| *cp < ',' as u32);
        font.codepoints.insert(at, ',' as u32);
        font.glyphs.insert(
            at,
            Glyph {
                x: 0,
                y: 0,
                width: 2,
                height: 3,
                bearing_x: 1,
                bearing_y: 2,
                advance: 3.0,
            },
        );
        let mut batch = UiBatch::new();
        font.text(&mut batch, 0.0, 0.0, ",", 1.0, [1.0; 4]);
        // x = bearing_x; y = ascent − bearing_y.
        assert_eq!(batch.quads[0].rect, [1.0, 5.0, 2.0, 3.0]);
        assert_eq!(font.ink_width(",", 1.0), 3.0);
        assert_eq!(font.width(",", 1.0), 3.0);
    }

    #[test]
    fn wrapping_fills_a_line_without_overflowing_it() {
        let font = mono();
        let text = "AAA BBB CCC AAA BBB";
        let lines = font.wrap(text, 2.0, 100.0);
        assert!(lines.len() > 1, "it did not wrap: {lines:?}");
        for line in &lines {
            assert!(font.ink_width(line, 2.0) <= 100.0, "{line:?} overflows");
        }
        // Nothing is lost and nothing is invented.
        assert_eq!(lines.join(" "), text);

        // A word wider than the line goes out on its own rather than looping.
        assert_eq!(font.wrap("AAAAAAAA", 4.0, 20.0), ["AAAAAAAA"]);
        assert!(font.wrap("", 2.0, 100.0).is_empty());
        assert!(font.wrap("   ", 2.0, 100.0).is_empty());
    }

    #[test]
    fn a_shape_ignores_the_baseline_and_fills_its_rect() {
        let font = mono();
        let mut batch = UiBatch::new();
        font.shape(&mut batch, [4.0, 5.0, 40.0, 40.0], 'A', [1.0; 4]);
        assert_eq!(batch.quads[0].rect, [4.0, 5.0, 40.0, 40.0]);
        assert_eq!(batch.quads[0].uv, font.uv(font.glyph('A').unwrap()));
        // A space and an uncovered character both draw nothing.
        font.shape(&mut batch, [0.0, 0.0, 8.0, 8.0], ' ', [1.0; 4]);
        font.shape(&mut batch, [0.0, 0.0, 8.0, 8.0], '—', [1.0; 4]);
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn an_asset_round_trips_and_expands_to_a_premultiplied_image() {
        let asset = FontAsset {
            width: 2,
            height: 1,
            coverage: vec![0, 255],
            sizes: vec![mono()],
        };
        let bytes = asset.to_bytes().expect("encode");
        let back = FontAsset::from_bytes(&bytes).expect("decode");
        assert_eq!(back.width, 2);
        assert_eq!(back.coverage, vec![0, 255]);
        // The handle is the game's, so it does not ride in the blob.
        assert_eq!(back.sizes[0].atlas, TextureHandle(0));

        let handle = TextureHandle(0x1234);
        let image = back.image(handle);
        assert!(image.is_valid());
        assert_eq!(image.handle, handle);
        assert_eq!(image.rgba, vec![0, 0, 0, 0, 255, 255, 255, 255]);

        let fonts = back.fonts(handle);
        assert_eq!(fonts[0].atlas, handle);
        assert_eq!((fonts[0].atlas_width, fonts[0].atlas_height), (2, 1));
    }

    #[test]
    fn a_malformed_asset_is_an_error_rather_than_a_panic() {
        let bad = FontAsset {
            width: 2,
            height: 2,
            coverage: vec![0],
            sizes: vec![mono()],
        };
        let bytes = bad.to_bytes().expect("encode");
        assert!(matches!(
            FontAsset::from_bytes(&bytes),
            Err(FontError::Malformed(_))
        ));
        assert!(matches!(
            FontAsset::from_bytes(&[0xff, 0xff, 0xff]),
            Err(FontError::Decode(_))
        ));

        let mut unsorted = mono();
        unsorted.codepoints.reverse();
        let asset = FontAsset {
            width: 1,
            height: 1,
            coverage: vec![0],
            sizes: vec![unsorted],
        };
        let bytes = asset.to_bytes().expect("encode");
        assert!(matches!(
            FontAsset::from_bytes(&bytes),
            Err(FontError::Malformed(_))
        ));
    }

    #[test]
    fn the_nearest_size_is_the_one_that_needs_no_resampling() {
        let at = |design_scale: f32| BitmapFont {
            design_scale,
            ..mono()
        };
        let sizes = vec![at(1.0), at(2.0), at(4.0), at(6.0)];
        let pick = |scale| nearest(&sizes, scale).design_scale;
        assert_eq!(pick(1.0), 1.0);
        assert_eq!(pick(2.0), 2.0);
        assert_eq!(pick(4.0), 4.0);
        assert_eq!(pick(9.0), 6.0);
        // A tie picks the larger: shrinking reads better than growing.
        assert_eq!(pick(3.0), 4.0);
        assert_eq!(pick(1.5), 2.0);
    }
}
