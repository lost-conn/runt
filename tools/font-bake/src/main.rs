//! `font-bake` — a TTF/OTF and a list of pixel sizes in, a
//! [`FontAsset`](runt_core::font::FontAsset) out.
//!
//! ```text
//! cargo run -p font-bake -- \
//!     --font assets/fonts/NotoSans-Regular.ttf \
//!     --out  assets/fonts/ui.font \
//!     --size 1:12 --size 2:24 --size 3:36 --size 4:48 --size 6:72 \
//!     --ascii --chars '△↑↓←→●▸' --fallback-micro
//! ```
//!
//! # Why this is a tool and not a runtime
//!
//! Because a font file is a decoder, a rasterizer and an atlas packer, and the
//! argument runt has always made against carrying those is an argument about the
//! *runtime*, not about the pixels. So they live here, in a binary that is a
//! workspace member and deliberately not a default member. It runs by hand, its
//! output is a file somebody commits, and nothing that ships links it.
//!
//! # One atlas, one rasterization per size
//!
//! The UI sampler is `Nearest` with `filterable: false` — deliberately, so the
//! pass stays valid under `downlevel_webgl2_defaults` — and a HUD draws at fixed
//! integer scales. So every size is fitted to its own pixel grid rather than
//! resampled from one master, which is the difference between crisp text and a
//! blur. They share **one** atlas image, because
//! `TextureRegistry::insert_image` is idempotent by handle: one handle is one
//! set of pixels forever, so N images would mean N handles and N draw calls.
//!
//! # Determinism
//!
//! Same input bytes → byte-identical output, because headless screenshot tests
//! depend on stable pixels. Everything here is ordered explicitly: glyphs are
//! visited in codepoint order, the packer sorts by `(height, codepoint, size)`
//! with no ties left to chance, there is no hash map anywhere on the path, and
//! coverage is quantized with a fixed rounding rule. Re-running the command on
//! an unchanged font produces a file `cmp` cannot tell from the last one.
//!
//! # Missing codepoints
//!
//! A typeface is allowed not to contain `▸`. `--fallback-micro` substitutes
//! `runt_core::font::micro`'s hand-authored 8 × 8 cell for any requested
//! codepoint the face has no glyph for, upscaled by the integer factor the size
//! implies — a hand-authored arrow beats a hole, and both beat a `.notdef` box.
//! Without the flag a missing codepoint is an error, because silently dropping
//! one is how a menu label grows a gap nobody notices until a screenshot.

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::process::ExitCode;

use ab_glyph::{Font, FontVec, PxScale, ScaleFont};
use runt_core::font::{micro, BitmapFont, FontAsset, Glyph};

// ---------------------------------------------------------------------------
// Arguments
// ---------------------------------------------------------------------------

const USAGE: &str = "\
font-bake — TTF/OTF → a runt FontAsset (one atlas, one glyph table per size)

USAGE:
    font-bake --font <FILE> --out <FILE> --size <SCALE>:<PX> [OPTIONS]

REQUIRED:
    --font <FILE>        the typeface to rasterize
    --out <FILE>         where to write the postcard blob
    --size <SCALE>:<PX>  bake the font a call site asks for as `scale: SCALE`
                         at PX raster pixels; repeat for more than one size.
                         PX is whatever ab_glyph means by a size, which for a
                         TTF is the ascender-to-descender span, NOT the em —
                         run once and read the reported cap height.

CHARSET (at least one required):
    --ascii              printable ASCII, U+0020..=U+007E
    --latin1             --ascii plus U+00A0..=U+00FF
    --chars <STR>        every character in STR
    --range <LO>-<HI>    an inclusive codepoint range: 32-126, 0x20-0x7e, U+20-U+7E

OPTIONS:
    --padding <N>        texels of gutter between packed glyphs [default: 1]
    --atlas-width <N>    force the atlas width instead of choosing the tightest
    --fallback-micro     substitute the 8x8 micro-font for codepoints the face
                         does not contain, instead of failing
    --no-kerning         drop the kerning table even if the face has one
    --quiet              print nothing on success
    -h, --help           this
";

struct Args {
    font: PathBuf,
    out: PathBuf,
    /// `(design scale, raster px)`, sorted by design scale.
    sizes: Vec<(f32, u32)>,
    charset: Vec<char>,
    padding: u32,
    atlas_width: Option<u32>,
    fallback_micro: bool,
    kerning: bool,
    quiet: bool,
}

/// Parse `--range`'s `LO-HI`, in decimal, `0x…` or `U+…`.
fn parse_codepoint(text: &str) -> Result<u32, String> {
    let trimmed = text.trim();
    let (body, radix) = if let Some(hex) = trimmed
        .strip_prefix("U+")
        .or_else(|| trimmed.strip_prefix("u+"))
        .or_else(|| trimmed.strip_prefix("0x"))
        .or_else(|| trimmed.strip_prefix("0X"))
    {
        (hex, 16)
    } else {
        (trimmed, 10)
    };
    u32::from_str_radix(body, radix).map_err(|_| format!("{text:?} is not a codepoint"))
}

fn parse_range(text: &str) -> Result<Vec<char>, String> {
    // Split on the last '-' that is not the first character, so `U+2190-U+2193`
    // works and a lone `-` is still an error rather than an empty range.
    let at = text
        .char_indices()
        .skip(1)
        .find(|(_, c)| *c == '-')
        .map(|(i, _)| i)
        .ok_or_else(|| format!("{text:?} is not a LO-HI range"))?;
    let lo = parse_codepoint(&text[..at])?;
    let hi = parse_codepoint(&text[at + 1..])?;
    if hi < lo {
        return Err(format!("{text:?} runs backwards"));
    }
    Ok((lo..=hi).filter_map(char::from_u32).collect())
}

/// Parse `--size`'s `SCALE:PX`.
fn parse_size(text: &str) -> Result<(f32, u32), String> {
    let (scale, px) = text
        .split_once(':')
        .ok_or_else(|| format!("--size {text:?} is not SCALE:PX"))?;
    let scale: f32 = scale
        .trim()
        .parse()
        .map_err(|_| format!("--size {text:?}: {scale:?} is not a scale"))?;
    let px: u32 = px
        .trim()
        .parse()
        .map_err(|_| format!("--size {text:?}: {px:?} is not a pixel size"))?;
    if !(scale.is_finite() && scale > 0.0) {
        return Err(format!("--size {text:?}: a scale must be positive"));
    }
    if px == 0 {
        return Err(format!("--size {text:?}: zero pixels has no glyphs"));
    }
    Ok((scale, px))
}

fn parse_args(argv: impl Iterator<Item = String>) -> Result<Option<Args>, String> {
    let mut font = None;
    let mut out = None;
    let mut sizes: Vec<(f32, u32)> = Vec::new();
    let mut charset: Vec<char> = Vec::new();
    let mut padding = 1u32;
    let mut atlas_width = None;
    let mut fallback_micro = false;
    let mut kerning = true;
    let mut quiet = false;

    let mut rest: Vec<String> = argv.collect();
    rest.reverse();
    let mut next = || rest.pop();

    while let Some(arg) = next() {
        let want = |arg: &str, got: Option<String>| -> Result<String, String> {
            got.ok_or_else(|| format!("{arg} wants a value"))
        };
        match arg.as_str() {
            "-h" | "--help" => return Ok(None),
            "--font" => font = Some(PathBuf::from(want(&arg, next())?)),
            "--out" => out = Some(PathBuf::from(want(&arg, next())?)),
            "--size" => {
                let raw = want(&arg, next())?;
                let size = parse_size(&raw)?;
                if sizes.iter().any(|(s, _)| *s == size.0) {
                    return Err(format!("--size {raw:?} names a scale already baked"));
                }
                sizes.push(size);
            }
            "--ascii" => charset.extend(' '..='~'),
            "--latin1" => {
                charset.extend(' '..='~');
                charset.extend('\u{a0}'..='\u{ff}');
            }
            "--chars" => charset.extend(want(&arg, next())?.chars()),
            "--range" => charset.extend(parse_range(&want(&arg, next())?)?),
            "--padding" => {
                let raw = want(&arg, next())?;
                padding = raw
                    .parse()
                    .map_err(|_| format!("--padding {raw:?} is not a number"))?;
            }
            "--atlas-width" => {
                let raw = want(&arg, next())?;
                let w: u32 = raw
                    .parse()
                    .map_err(|_| format!("--atlas-width {raw:?} is not a number"))?;
                atlas_width = Some(w);
            }
            "--fallback-micro" => fallback_micro = true,
            "--no-kerning" => kerning = false,
            "--quiet" => quiet = true,
            other => return Err(format!("unknown argument {other:?}")),
        }
    }

    let font = font.ok_or("--font is required")?;
    let out = out.ok_or("--out is required")?;
    if sizes.is_empty() {
        return Err("at least one --size is required".into());
    }
    if charset.is_empty() {
        return Err("a charset is required: --ascii, --latin1, --chars or --range".into());
    }
    sizes.sort_by(|a, b| a.0.total_cmp(&b.0));
    // Sorted and deduplicated: the codepoint map is a sorted `Vec` and the
    // charset is where that starts.
    charset.sort_unstable();
    charset.dedup();

    Ok(Some(Args {
        font,
        out,
        sizes,
        charset,
        padding,
        atlas_width,
        fallback_micro,
        kerning,
        quiet,
    }))
}

// ---------------------------------------------------------------------------
// Rasterization
// ---------------------------------------------------------------------------

/// One rasterized glyph, before it has been given a place in the atlas.
struct Raster {
    codepoint: u32,
    width: u32,
    height: u32,
    bearing_x: i32,
    /// Baseline → ink top, positive up. Filled in for face glyphs; for a micro
    /// fallback it is the font's ascent and therefore only known once the face
    /// glyphs have been measured.
    bearing_y: i32,
    advance: f32,
    /// `width · height` bytes of coverage.
    coverage: Vec<u8>,
    /// Did this come from the micro-font rather than the typeface?
    fallback: bool,
}

/// Coverage → a byte, with one fixed rounding rule so two runs agree.
fn quantize(coverage: f32) -> u8 {
    let c = if coverage.is_nan() {
        0.0
    } else {
        coverage.clamp(0.0, 1.0)
    };
    (c * 255.0 + 0.5) as u8
}

/// Rasterize one size of one face. Returns the glyphs in codepoint order.
fn rasterize(
    face: &FontVec,
    scale: f32,
    px: u32,
    charset: &[char],
    fallback_micro: bool,
) -> Result<(Vec<Raster>, Vec<char>), String> {
    let scaled = face.as_scaled(PxScale::from(px as f32));
    let mut glyphs = Vec::with_capacity(charset.len());
    let mut missing = Vec::new();

    for &c in charset {
        let id = face.glyph_id(c);
        // ab_glyph maps an absent character to glyph 0 (`.notdef`), which draws
        // a box. A box is a wrong glyph, and a wrong glyph is worse than a gap.
        if id.0 == 0 {
            missing.push(c);
            continue;
        }
        let advance = scaled.h_advance(id);
        let positioned =
            id.with_scale_and_position(PxScale::from(px as f32), ab_glyph::point(0.0, 0.0));
        match face.outline_glyph(positioned) {
            // A space has an advance and no outline. That is not an error, it is
            // a space: zero-size ink, which every drawing path skips.
            None => glyphs.push(Raster {
                codepoint: c as u32,
                width: 0,
                height: 0,
                bearing_x: 0,
                bearing_y: 0,
                advance,
                coverage: Vec::new(),
                fallback: false,
            }),
            Some(outlined) => {
                // `px_bounds` is integer-aligned (floor/ceil), so these are exact
                // texel counts rather than a rounding decision made here.
                let bounds = outlined.px_bounds();
                let width = (bounds.max.x - bounds.min.x) as u32;
                let height = (bounds.max.y - bounds.min.y) as u32;
                let mut coverage = vec![0u8; (width * height) as usize];
                outlined.draw(|x, y, c| {
                    if x < width && y < height {
                        coverage[(y * width + x) as usize] = quantize(c);
                    }
                });
                glyphs.push(Raster {
                    codepoint: c as u32,
                    width,
                    height,
                    bearing_x: bounds.min.x as i32,
                    bearing_y: -bounds.min.y as i32,
                    advance,
                    coverage,
                    fallback: false,
                });
            }
        }
    }

    if !missing.is_empty() && !fallback_micro {
        let list: String = missing.iter().collect();
        return Err(format!(
            "the face has no glyph for {list:?} at {px} px; pass --fallback-micro to substitute \
             the 8x8 micro-font, or drop them from the charset"
        ));
    }

    // A substituted cell is scaled by the *design scale*, not by `px`: the
    // micro-font's cell is `UNIT` logical pixels at scale 1 by construction, so
    // `scale` is exactly its integer upscale — and nearest-neighbour is only
    // honest at whole numbers.
    if !missing.is_empty() && (scale.fract() != 0.0 || scale < 1.0) {
        let list: String = missing.iter().collect();
        return Err(format!(
            "scale {scale} is not a whole multiple of the micro-font's cell, so {list:?} cannot be \
             substituted without resampling"
        ));
    }
    let factor = scale as u32;
    for &c in &missing {
        let cell = micro::cell_of(c).ok_or_else(|| {
            format!("the micro-font has no cell for {c:?} either; drop it from the charset")
        })?;
        let bits = micro::cell_coverage(cell);
        // The 5 × 7 body, or the whole cell for the one glyph that is a shape.
        let (cw, ch) = if c == micro::DISC {
            (micro::CELL, micro::CELL)
        } else {
            (micro::GLYPH_W, micro::GLYPH_H)
        };
        let (width, height) = (cw * factor, ch * factor);
        let mut coverage = vec![0u8; (width * height) as usize];
        for y in 0..height {
            for x in 0..width {
                let src = (y / factor) * micro::CELL + (x / factor);
                coverage[(y * width + x) as usize] = bits[src as usize];
            }
        }
        glyphs.push(Raster {
            codepoint: c as u32,
            width,
            height,
            bearing_x: 0,
            // Patched to the font's ascent once every face glyph is measured, so
            // a substituted arrow tops out on the same line the letters do.
            bearing_y: 0,
            advance: (micro::ADVANCE * factor) as f32,
            coverage,
            fallback: true,
        });
    }

    glyphs.sort_by_key(|g| g.codepoint);
    Ok((glyphs, missing))
}

// ---------------------------------------------------------------------------
// Packing
// ---------------------------------------------------------------------------

/// Where one glyph landed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Placed {
    key: (usize, u32),
    x: u32,
    y: u32,
}

/// A shelf packer: rows of a fixed width, each as tall as its tallest glyph.
///
/// Dumb on purpose. A skyline packer would save maybe 10% of a texture that is
/// already a quarter of a megabyte, in exchange for a second thing that can be
/// subtly wrong; a shelf packer fed height-sorted input is within a few percent
/// of it and fits on a screen.
///
/// The sort is total — `(height desc, size index, codepoint)` — so there is no
/// tie for the platform's sort stability to decide.
fn pack(
    boxes: &[((usize, u32), u32, u32)],
    width: u32,
    padding: u32,
) -> Option<(Vec<Placed>, u32)> {
    let mut order: Vec<usize> = (0..boxes.len()).filter(|i| boxes[*i].1 > 0).collect();
    order.sort_by(|a, b| {
        let (ka, wa, ha) = boxes[*a];
        let (kb, wb, hb) = boxes[*b];
        hb.cmp(&ha)
            .then(wb.cmp(&wa))
            .then(ka.0.cmp(&kb.0))
            .then(ka.1.cmp(&kb.1))
    });

    let mut placed = Vec::with_capacity(order.len());
    let mut pen_x = padding;
    let mut shelf_y = padding;
    let mut shelf_h = 0u32;
    for i in order {
        let (key, w, h) = boxes[i];
        if w + padding * 2 > width {
            return None;
        }
        if pen_x + w + padding > width {
            shelf_y += shelf_h + padding;
            shelf_h = 0;
            pen_x = padding;
        }
        placed.push(Placed {
            key,
            x: pen_x,
            y: shelf_y,
        });
        pen_x += w + padding;
        shelf_h = shelf_h.max(h);
    }
    let height = shelf_y + shelf_h + padding;
    Some((placed, height))
}

/// WebGL2's downlevel `max_texture_dimension_2d`. DESIGN pillar 3: the baseline
/// fits the downlevel limits, and an atlas that does not is an atlas that only
/// fails in a browser.
const MAX_ATLAS: u32 = 2048;

/// The atlas width that packs these boxes into the least area.
fn choose_width(
    boxes: &[((usize, u32), u32, u32)],
    padding: u32,
) -> Option<(u32, Vec<Placed>, u32)> {
    let mut best: Option<(u32, Vec<Placed>, u32)> = None;
    for width in [128u32, 256, 512, 1024, 2048] {
        let Some((placed, height)) = pack(boxes, width, padding) else {
            continue;
        };
        if height > MAX_ATLAS {
            continue;
        }
        let area = width as u64 * height as u64;
        let better = best
            .as_ref()
            .is_none_or(|(w, _, h)| area < *w as u64 * *h as u64);
        if better {
            best = Some((width, placed, height));
        }
    }
    best
}

// ---------------------------------------------------------------------------
// The bake
// ---------------------------------------------------------------------------

struct Report {
    asset: FontAsset,
    /// The codepoints that came from the micro-font instead of the face,
    /// sorted. The set is the same at every size — a face either has a glyph or
    /// it does not — so it is reported once.
    fallbacks: Vec<char>,
}

fn bake(args: &Args, face: &FontVec) -> Result<Report, String> {
    let mut per_size: Vec<Vec<Raster>> = Vec::with_capacity(args.sizes.len());
    let mut boxes_of: Vec<(i32, i32)> = Vec::with_capacity(args.sizes.len());
    let mut fallbacks: BTreeSet<char> = BTreeSet::new();
    for &(scale, px) in &args.sizes {
        let (mut glyphs, missing) = rasterize(face, scale, px, &args.charset, args.fallback_micro)?;
        // The line box, measured from the ink **the face** puts in this charset:
        // not from `hhea` (whose ascender reserves room for accents no ASCII
        // string contains) and not from the substituted cells (an 8 × 8 disc is
        // a shape, and letting it set the leading would push every paragraph
        // apart to make room for a health dot).
        let face_ink = || glyphs.iter().filter(|g| !g.fallback && g.height > 0);
        let ascent = face_ink().map(|g| g.bearing_y).max().unwrap_or(px as i32);
        let descent = face_ink()
            .map(|g| g.height as i32 - g.bearing_y)
            .max()
            .unwrap_or(0);
        // A substituted cell tops out on the same line the letters do.
        for glyph in glyphs.iter_mut().filter(|g| g.fallback) {
            glyph.bearing_y = ascent;
        }
        fallbacks.extend(missing);
        boxes_of.push((ascent, descent));
        per_size.push(glyphs);
    }

    // One box list across every size, so the sizes interleave on the shelves
    // instead of each starting a new one.
    let mut boxes: Vec<((usize, u32), u32, u32)> = Vec::new();
    for (size, glyphs) in per_size.iter().enumerate() {
        for glyph in glyphs {
            boxes.push(((size, glyph.codepoint), glyph.width, glyph.height));
        }
    }

    let (width, placed, height) = match args.atlas_width {
        Some(width) => {
            let (placed, height) = pack(&boxes, width, args.padding)
                .ok_or_else(|| format!("a glyph is wider than the {width}-texel atlas"))?;
            if height > MAX_ATLAS {
                return Err(format!(
                    "the atlas comes out {width}×{height}, past WebGL2's {MAX_ATLAS} limit"
                ));
            }
            (width, placed, height)
        }
        None => choose_width(&boxes, args.padding)
            .ok_or_else(|| format!("these glyphs do not fit in a {MAX_ATLAS}×{MAX_ATLAS} atlas"))?,
    };

    let mut coverage = vec![0u8; (width * height) as usize];
    // Sorted so the blit order is a property of the data, not of the packer.
    let mut placed = placed;
    placed.sort_by_key(|p| p.key);
    let mut at: Vec<Option<Placed>> = vec![None; boxes.len()];
    for (index, ((size, codepoint), _, _)) in boxes.iter().enumerate() {
        at[index] = placed
            .binary_search_by_key(&(*size, *codepoint), |p| p.key)
            .ok()
            .map(|i| placed[i]);
    }

    let mut sizes = Vec::with_capacity(args.sizes.len());
    let mut cursor = 0usize;
    for (index, &(scale, px)) in args.sizes.iter().enumerate() {
        let glyphs = &per_size[index];
        let (ascent, descent) = boxes_of[index];
        let scaled = face.as_scaled(PxScale::from(px as f32));
        let mut table = Vec::with_capacity(glyphs.len());
        let mut codepoints = Vec::with_capacity(glyphs.len());
        for glyph in glyphs {
            let spot = at[cursor];
            cursor += 1;
            let (x, y) = match spot {
                Some(p) => (p.x, p.y),
                None => (0, 0),
            };
            for row in 0..glyph.height {
                let dst = ((y + row) * width + x) as usize;
                let src = (row * glyph.width) as usize;
                coverage[dst..dst + glyph.width as usize]
                    .copy_from_slice(&glyph.coverage[src..src + glyph.width as usize]);
            }
            codepoints.push(glyph.codepoint);
            table.push(Glyph {
                x: x as u16,
                y: y as u16,
                width: glyph.width as u16,
                height: glyph.height as u16,
                bearing_x: glyph.bearing_x as i16,
                bearing_y: glyph.bearing_y as i16,
                advance: glyph.advance,
            });
        }

        let kerning = if args.kerning {
            let mut pairs = Vec::new();
            for left in &codepoints {
                for right in &codepoints {
                    let (a, b) = (
                        face.glyph_id(char::from_u32(*left).unwrap()),
                        face.glyph_id(char::from_u32(*right).unwrap()),
                    );
                    if a.0 == 0 || b.0 == 0 {
                        continue;
                    }
                    let adjust = scaled.kern(a, b);
                    if adjust != 0.0 {
                        pairs.push(runt_core::font::Kern {
                            left: *left,
                            right: *right,
                            adjust,
                        });
                    }
                }
            }
            pairs
        } else {
            Vec::new()
        };

        let space = table
            .iter()
            .zip(&codepoints)
            .find(|(_, cp)| **cp == ' ' as u32)
            .map(|(g, _)| g.advance)
            .unwrap_or(px as f32 * 0.25);

        sizes.push(BitmapFont {
            atlas_width: width,
            atlas_height: height,
            px: px as f32,
            design_scale: scale,
            ascent: ascent as f32,
            descent: descent as f32,
            // The ink's own box. A face's declared `hhea` line height reserves
            // room for accents and a leading this HUD does not draw; a caller
            // that wants air adds it, and every one of them already does.
            line_height: (ascent + descent) as f32,
            missing_advance: space,
            glyphs: table,
            codepoints,
            kerning,
            ..BitmapFont::default()
        });
    }

    Ok(Report {
        asset: FontAsset {
            width,
            height,
            coverage,
            sizes,
        },
        fallbacks: fallbacks.into_iter().collect(),
    })
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn run() -> Result<(), String> {
    let args = match parse_args(std::env::args().skip(1))? {
        None => {
            print!("{USAGE}");
            return Ok(());
        }
        Some(args) => args,
    };

    let bytes = std::fs::read(&args.font)
        .map_err(|e| format!("cannot read {}: {e}", args.font.display()))?;
    let face = FontVec::try_from_vec(bytes).map_err(|e| {
        format!(
            "{} is not a font this tool can read: {e}",
            args.font.display()
        )
    })?;

    let report = bake(&args, &face)?;
    let blob = report
        .asset
        .to_bytes()
        .map_err(|e| format!("the asset did not encode: {e}"))?;
    // Round-trips through the loader a game will use, here rather than in the
    // game: a blob that fails `from_bytes` should never reach a commit.
    FontAsset::from_bytes(&blob).map_err(|e| format!("the asset does not load back: {e}"))?;

    if let Some(dir) = args.out.parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("cannot make {}: {e}", dir.display()))?;
    }
    std::fs::write(&args.out, &blob)
        .map_err(|e| format!("cannot write {}: {e}", args.out.display()))?;

    if !args.quiet {
        let asset = &report.asset;
        println!("{} → {}", args.font.display(), args.out.display());
        println!(
            "  atlas    {}×{} = {} texels ({} KiB of coverage, {} KiB uploaded as RGBA8)",
            asset.width,
            asset.height,
            asset.width * asset.height,
            asset.coverage.len() / 1024,
            asset.coverage.len() * 4 / 1024,
        );
        println!("  blob     {} bytes", blob.len());
        for font in &asset.sizes {
            // Cap height is what a designer's eye compares, so report it: it is
            // the number to turn `--size`'s PX against when the text comes out
            // the wrong size next to a HUD that was drawn for a bitmap font.
            let cap = font
                .glyph('H')
                .map(|g| g.bearing_y)
                .unwrap_or(font.ascent as i16);
            println!(
                "  scale {:<4} {:>3} px   cap {:>3}  ascent {:>3}  descent {:>3}  line {:>3}  \
                 {} glyphs, {} kerning pairs",
                font.design_scale,
                font.px,
                cap,
                font.ascent,
                font.descent,
                font.line_height,
                font.glyphs.len(),
                font.kerning.len(),
            );
        }
        if !report.fallbacks.is_empty() {
            let list: String = report.fallbacks.iter().collect();
            println!(
                "  fallback the face has no glyph for {list:?}; the 8x8 micro-font's \
                 cells were substituted at every size"
            );
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("font-bake: {e}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Result<Option<Args>, String> {
        parse_args(list.iter().map(|s| s.to_string()))
    }

    #[test]
    fn the_charset_flags_add_up_and_come_out_sorted() {
        let parsed = args(&[
            "--font",
            "f.ttf",
            "--out",
            "o.font",
            "--size",
            "2:22",
            "--ascii",
            "--chars",
            "→△",
            "--range",
            "U+2190-U+2192",
        ])
        .unwrap()
        .unwrap();
        assert_eq!(parsed.charset[0], ' ');
        assert!(parsed.charset.windows(2).all(|w| w[0] < w[1]), "not sorted");
        for c in [' ', '~', '△', '←', '↑', '→'] {
            assert!(parsed.charset.contains(&c), "{c:?} missing");
        }
        // `--chars →` and `--range …2192` name the same character once.
        assert_eq!(parsed.charset.iter().filter(|c| **c == '→').count(), 1);
    }

    #[test]
    fn a_missing_required_argument_is_an_error_rather_than_a_default() {
        assert!(args(&["--out", "o", "--size", "2:22", "--ascii"]).is_err());
        assert!(args(&["--font", "f", "--size", "2:22", "--ascii"]).is_err());
        assert!(args(&["--font", "f", "--out", "o", "--ascii"]).is_err());
        assert!(args(&["--font", "f", "--out", "o", "--size", "2:22"]).is_err());
        assert!(args(&["--nonsense"]).is_err());
        assert!(args(&["--help"]).unwrap().is_none());

        // A size is a scale *and* a raster size: neither half is guessable.
        assert!(parse_size("22").is_err());
        assert!(parse_size("2:0").is_err());
        assert!(parse_size("0:22").is_err());
        assert!(parse_size("x:22").is_err());
        assert_eq!(parse_size("2:22"), Ok((2.0, 22)));
        // …and no scale may be baked twice, or `nearest` picks by luck.
        assert!(args(&[
            "--font", "f", "--out", "o", "--ascii", "--size", "2:22", "--size", "2:24"
        ])
        .is_err());
    }

    #[test]
    fn codepoints_parse_in_the_three_spellings_people_write() {
        assert_eq!(parse_codepoint("32"), Ok(32));
        assert_eq!(parse_codepoint("0x7e"), Ok(126));
        assert_eq!(parse_codepoint("U+2190"), Ok(0x2190));
        assert!(parse_codepoint("nope").is_err());
        assert_eq!(parse_range("32-34").unwrap(), [' ', '!', '"']);
        assert!(parse_range("34-32").is_err());
        assert!(parse_range("32").is_err());
    }

    #[test]
    fn the_packer_is_a_function_of_its_input_and_nothing_else() {
        let boxes: Vec<((usize, u32), u32, u32)> = (0..64)
            .map(|i| {
                (
                    (i % 3, 32 + i as u32),
                    3 + (i as u32 * 7) % 11,
                    4 + (i as u32 * 5) % 9,
                )
            })
            .collect();
        let (a, ha) = pack(&boxes, 64, 1).unwrap();
        let (b, hb) = pack(&boxes, 64, 1).unwrap();
        assert_eq!(a, b);
        assert_eq!(ha, hb);

        // Nothing overlaps and nothing leaves the atlas.
        let by_key: Vec<_> = boxes.iter().map(|(k, w, h)| (*k, *w, *h)).collect();
        for spot in &a {
            let (_, w, h) = by_key.iter().find(|(k, _, _)| *k == spot.key).unwrap();
            assert!(spot.x + w <= 64, "{spot:?} leaves the atlas");
            assert!(spot.y + h <= ha, "{spot:?} leaves the atlas");
            for other in &a {
                if other.key == spot.key {
                    continue;
                }
                let (_, ow, oh) = by_key.iter().find(|(k, _, _)| *k == other.key).unwrap();
                let apart = spot.x + w <= other.x
                    || other.x + ow <= spot.x
                    || spot.y + h <= other.y
                    || other.y + oh <= spot.y;
                assert!(apart, "{spot:?} overlaps {other:?}");
            }
        }
    }

    #[test]
    fn a_glyph_wider_than_the_atlas_fails_rather_than_wrapping() {
        assert!(pack(&[((0, 65), 200, 8)], 64, 1).is_none());
        // Zero-width boxes — spaces — are placed nowhere and cost nothing.
        let (placed, height) = pack(&[((0, 32), 0, 0)], 64, 1).unwrap();
        assert!(placed.is_empty());
        assert_eq!(height, 2);
    }

    #[test]
    fn coverage_quantizes_the_same_way_every_time() {
        assert_eq!(quantize(0.0), 0);
        assert_eq!(quantize(1.0), 255);
        assert_eq!(quantize(0.5), 128);
        // Out of range and NaN are clamped rather than wrapped, because a
        // rasterizer that hands back 1.0000001 must not produce a black pixel.
        assert_eq!(quantize(-1.0), 0);
        assert_eq!(quantize(2.0), 255);
        assert_eq!(quantize(f32::NAN), 0);
    }
}
