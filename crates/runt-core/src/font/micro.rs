//! The optional fallback typeface: 102 glyphs of 8 × 8, 1-bit — `default-font`.
//!
//! Off by default, and the **only** letters in this engine. A runt game is
//! expected to bake its own face with `tools/font-bake`; this exists so that the
//! hour before it picks one is not an hour with no text on the screen, and so
//! `tools/font-bake` has something to substitute when a real typeface turns out
//! not to contain `▸`.
//!
//! ```text
//! CELLS      102 glyphs: ASCII 32..=126, then △ ↑ ↓ ← → ● ▸
//! grid       16 × 7 cells of 8 × 8 texels  →  a 128 × 56 atlas
//! text       5 wide, 7 tall, drawn from the top-left of its cell
//! advance    6 texels at scale 1 (5 + a column of air)
//! ```
//!
//! # Why 8 × 8 and why a table
//!
//! Because at this budget it is not a compromise, it is the correct data
//! structure. A hundred glyphs at one size, on a HUD that is already
//! deliberately chunky, cost **816 bytes** of table here — against a decoder, a
//! rasterizer and an atlas packer for the alternative. Those exist now, in
//! `tools/font-bake`, but they run at *build* time and are not linked into
//! anything that ships; this table is what a game has before it has run them.
//! No allocation until [`asset`] is called, no I/O ever, and the same pixels on
//! every platform including the headless test that screenshots them.
//!
//! # The 5 × 7 body inside an 8 × 8 cell
//!
//! The cell is 8 wide so that [`DISC`] — the one glyph that is a *shape* rather
//! than a letter — can use the whole of it and still scale to a 14 px health dot
//! without a second mechanism. Text uses the left 5 columns and the top 7 rows;
//! the spare column is the letter-spacing and the spare row is the leading, so a
//! line lays out by stepping the pen and never has to know about kerning at all.

use super::{BitmapFont, FontAsset, Glyph, UNIT};

// ---------------------------------------------------------------------------
// The grid
// ---------------------------------------------------------------------------

/// Texels per cell, both axes.
pub const CELL: u32 = 8;
/// Cells per atlas row. 16 keeps the atlas 128 texels wide, which is a friendly
/// number for every backend and small enough to read in a hex dump.
pub const COLUMNS: u32 = 16;
/// How many cells the printable ASCII range fills: `' '` (32) through `'~'`
/// (126).
pub const ASCII_CELLS: usize = 95;
/// The first ASCII code point with a cell.
pub const ASCII_FIRST: u8 = 32;

/// Width of a text glyph's body, in texels. The remaining columns of the cell
/// are the letter-spacing.
pub const GLYPH_W: u32 = 5;
/// Height of a text glyph's body, in texels.
pub const GLYPH_H: u32 = 7;
/// Pen step between two characters, in texels at scale 1.
pub const ADVANCE: u32 = GLYPH_W + 1;

/// Atlas width in texels.
pub const fn atlas_width() -> u32 {
    COLUMNS * CELL
}

/// Atlas height in texels — however many rows the cells need.
pub const fn atlas_height() -> u32 {
    (CELLS.len() as u32).div_ceil(COLUMNS) * CELL
}

// ---------------------------------------------------------------------------
// The font
// ---------------------------------------------------------------------------

/// Every glyph, as eight rows of eight bits. Bit 7 is the leftmost column and
/// row 0 is the top.
///
/// Index 0..=94 is ASCII 32..=126 in order, so a character's cell is
/// `c - 32`. Past that are the shapes a HUD needs and ASCII has no code for —
/// see [`TRIANGLE`] and the constants below it.
#[rustfmt::skip]
pub const CELLS: &[[u8; 8]] = &[
    // 0   space
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    // 1   '!'
    [0x20, 0x20, 0x20, 0x20, 0x20, 0x00, 0x20, 0x00],
    // 2   '"'
    [0x50, 0x50, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    // 3   '#'
    [0x50, 0xf8, 0x50, 0x50, 0xf8, 0x50, 0x00, 0x00],
    // 4   '$'
    [0x20, 0x78, 0xa0, 0x70, 0x28, 0xf0, 0x20, 0x00],
    // 5   '%'
    [0xc8, 0xc8, 0x10, 0x20, 0x40, 0x98, 0x98, 0x00],
    // 6   '&'
    [0x60, 0x90, 0x90, 0x60, 0xa8, 0x90, 0x68, 0x00],
    // 7   "'"
    [0x20, 0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    // 8   '('
    [0x10, 0x20, 0x40, 0x40, 0x40, 0x20, 0x10, 0x00],
    // 9   ')'
    [0x40, 0x20, 0x10, 0x10, 0x10, 0x20, 0x40, 0x00],
    // 10  '*'
    [0x00, 0xa8, 0x70, 0xf8, 0x70, 0xa8, 0x00, 0x00],
    // 11  '+'
    [0x00, 0x20, 0x20, 0xf8, 0x20, 0x20, 0x00, 0x00],
    // 12  ','
    [0x00, 0x00, 0x00, 0x00, 0x30, 0x20, 0x40, 0x00],
    // 13  '-'
    [0x00, 0x00, 0x00, 0xf8, 0x00, 0x00, 0x00, 0x00],
    // 14  '.'
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x60, 0x60, 0x00],
    // 15  '/'
    [0x08, 0x10, 0x10, 0x20, 0x40, 0x40, 0x80, 0x00],
    // 16  '0'
    [0x70, 0x88, 0x98, 0xa8, 0xc8, 0x88, 0x70, 0x00],
    // 17  '1'
    [0x20, 0x60, 0x20, 0x20, 0x20, 0x20, 0x70, 0x00],
    // 18  '2'
    [0x70, 0x88, 0x08, 0x10, 0x20, 0x40, 0xf8, 0x00],
    // 19  '3'
    [0xf8, 0x10, 0x20, 0x10, 0x08, 0x88, 0x70, 0x00],
    // 20  '4'
    [0x10, 0x30, 0x50, 0x90, 0xf8, 0x10, 0x10, 0x00],
    // 21  '5'
    [0xf8, 0x80, 0xf0, 0x08, 0x08, 0x88, 0x70, 0x00],
    // 22  '6'
    [0x30, 0x40, 0x80, 0xf0, 0x88, 0x88, 0x70, 0x00],
    // 23  '7'
    [0xf8, 0x08, 0x10, 0x20, 0x40, 0x40, 0x40, 0x00],
    // 24  '8'
    [0x70, 0x88, 0x88, 0x70, 0x88, 0x88, 0x70, 0x00],
    // 25  '9'
    [0x70, 0x88, 0x88, 0x78, 0x08, 0x10, 0x60, 0x00],
    // 26  ':'
    [0x00, 0x60, 0x60, 0x00, 0x60, 0x60, 0x00, 0x00],
    // 27  ';'
    [0x00, 0x60, 0x60, 0x00, 0x60, 0x20, 0x40, 0x00],
    // 28  '<'
    [0x10, 0x20, 0x40, 0x80, 0x40, 0x20, 0x10, 0x00],
    // 29  '='
    [0x00, 0x00, 0xf8, 0x00, 0xf8, 0x00, 0x00, 0x00],
    // 30  '>'
    [0x40, 0x20, 0x10, 0x08, 0x10, 0x20, 0x40, 0x00],
    // 31  '?'
    [0x70, 0x88, 0x08, 0x10, 0x20, 0x00, 0x20, 0x00],
    // 32  '@'
    [0x70, 0x88, 0xb8, 0xa8, 0xb8, 0x80, 0x70, 0x00],
    // 33  'A'
    [0x20, 0x50, 0x88, 0x88, 0xf8, 0x88, 0x88, 0x00],
    // 34  'B'
    [0xf0, 0x88, 0x88, 0xf0, 0x88, 0x88, 0xf0, 0x00],
    // 35  'C'
    [0x70, 0x88, 0x80, 0x80, 0x80, 0x88, 0x70, 0x00],
    // 36  'D'
    [0xe0, 0x90, 0x88, 0x88, 0x88, 0x90, 0xe0, 0x00],
    // 37  'E'
    [0xf8, 0x80, 0x80, 0xf0, 0x80, 0x80, 0xf8, 0x00],
    // 38  'F'
    [0xf8, 0x80, 0x80, 0xf0, 0x80, 0x80, 0x80, 0x00],
    // 39  'G'
    [0x70, 0x88, 0x80, 0xb8, 0x88, 0x88, 0x78, 0x00],
    // 40  'H'
    [0x88, 0x88, 0x88, 0xf8, 0x88, 0x88, 0x88, 0x00],
    // 41  'I'
    [0x70, 0x20, 0x20, 0x20, 0x20, 0x20, 0x70, 0x00],
    // 42  'J'
    [0x38, 0x10, 0x10, 0x10, 0x10, 0x90, 0x60, 0x00],
    // 43  'K'
    [0x88, 0x90, 0xa0, 0xc0, 0xa0, 0x90, 0x88, 0x00],
    // 44  'L'
    [0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0xf8, 0x00],
    // 45  'M'
    [0x88, 0xd8, 0xa8, 0xa8, 0x88, 0x88, 0x88, 0x00],
    // 46  'N'
    [0x88, 0xc8, 0xa8, 0x98, 0x88, 0x88, 0x88, 0x00],
    // 47  'O'
    [0x70, 0x88, 0x88, 0x88, 0x88, 0x88, 0x70, 0x00],
    // 48  'P'
    [0xf0, 0x88, 0x88, 0xf0, 0x80, 0x80, 0x80, 0x00],
    // 49  'Q'
    [0x70, 0x88, 0x88, 0x88, 0xa8, 0x90, 0x68, 0x00],
    // 50  'R'
    [0xf0, 0x88, 0x88, 0xf0, 0xa0, 0x90, 0x88, 0x00],
    // 51  'S'
    [0x78, 0x80, 0x80, 0x70, 0x08, 0x08, 0xf0, 0x00],
    // 52  'T'
    [0xf8, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x00],
    // 53  'U'
    [0x88, 0x88, 0x88, 0x88, 0x88, 0x88, 0x70, 0x00],
    // 54  'V'
    [0x88, 0x88, 0x88, 0x88, 0x88, 0x50, 0x20, 0x00],
    // 55  'W'
    [0x88, 0x88, 0x88, 0xa8, 0xa8, 0xd8, 0x88, 0x00],
    // 56  'X'
    [0x88, 0x88, 0x50, 0x20, 0x50, 0x88, 0x88, 0x00],
    // 57  'Y'
    [0x88, 0x88, 0x50, 0x20, 0x20, 0x20, 0x20, 0x00],
    // 58  'Z'
    [0xf8, 0x08, 0x10, 0x20, 0x40, 0x80, 0xf8, 0x00],
    // 59  '['
    [0x70, 0x40, 0x40, 0x40, 0x40, 0x40, 0x70, 0x00],
    // 60  '\\'
    [0x80, 0x40, 0x40, 0x20, 0x10, 0x10, 0x08, 0x00],
    // 61  ']'
    [0x70, 0x10, 0x10, 0x10, 0x10, 0x10, 0x70, 0x00],
    // 62  '^'
    [0x20, 0x50, 0x88, 0x00, 0x00, 0x00, 0x00, 0x00],
    // 63  '_'
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf8, 0x00],
    // 64  '`'
    [0x40, 0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    // 65  'a'
    [0x00, 0x00, 0x70, 0x08, 0x78, 0x88, 0x78, 0x00],
    // 66  'b'
    [0x80, 0x80, 0xf0, 0x88, 0x88, 0x88, 0xf0, 0x00],
    // 67  'c'
    [0x00, 0x00, 0x70, 0x80, 0x80, 0x80, 0x70, 0x00],
    // 68  'd'
    [0x08, 0x08, 0x78, 0x88, 0x88, 0x88, 0x78, 0x00],
    // 69  'e'
    [0x00, 0x00, 0x70, 0x88, 0xf8, 0x80, 0x70, 0x00],
    // 70  'f'
    [0x30, 0x40, 0x40, 0xf0, 0x40, 0x40, 0x40, 0x00],
    // 71  'g'
    [0x00, 0x78, 0x88, 0x88, 0x78, 0x08, 0x70, 0x00],
    // 72  'h'
    [0x80, 0x80, 0xf0, 0x88, 0x88, 0x88, 0x88, 0x00],
    // 73  'i'
    [0x20, 0x00, 0x60, 0x20, 0x20, 0x20, 0x70, 0x00],
    // 74  'j'
    [0x10, 0x00, 0x30, 0x10, 0x10, 0x90, 0x60, 0x00],
    // 75  'k'
    [0x80, 0x80, 0x90, 0xa0, 0xc0, 0xa0, 0x90, 0x00],
    // 76  'l'
    [0x60, 0x20, 0x20, 0x20, 0x20, 0x20, 0x70, 0x00],
    // 77  'm'
    [0x00, 0x00, 0xd0, 0xa8, 0xa8, 0xa8, 0xa8, 0x00],
    // 78  'n'
    [0x00, 0x00, 0xf0, 0x88, 0x88, 0x88, 0x88, 0x00],
    // 79  'o'
    [0x00, 0x00, 0x70, 0x88, 0x88, 0x88, 0x70, 0x00],
    // 80  'p'
    [0x00, 0xf0, 0x88, 0x88, 0xf0, 0x80, 0x80, 0x00],
    // 81  'q'
    [0x00, 0x78, 0x88, 0x88, 0x78, 0x08, 0x08, 0x00],
    // 82  'r'
    [0x00, 0x00, 0xb0, 0xc0, 0x80, 0x80, 0x80, 0x00],
    // 83  's'
    [0x00, 0x00, 0x78, 0x80, 0x70, 0x08, 0xf0, 0x00],
    // 84  't'
    [0x40, 0x40, 0xf0, 0x40, 0x40, 0x48, 0x30, 0x00],
    // 85  'u'
    [0x00, 0x00, 0x88, 0x88, 0x88, 0x98, 0x68, 0x00],
    // 86  'v'
    [0x00, 0x00, 0x88, 0x88, 0x88, 0x50, 0x20, 0x00],
    // 87  'w'
    [0x00, 0x00, 0x88, 0xa8, 0xa8, 0xa8, 0x50, 0x00],
    // 88  'x'
    [0x00, 0x00, 0x88, 0x50, 0x20, 0x50, 0x88, 0x00],
    // 89  'y'
    [0x00, 0x88, 0x88, 0x88, 0x78, 0x08, 0x70, 0x00],
    // 90  'z'
    [0x00, 0x00, 0xf8, 0x10, 0x20, 0x40, 0xf8, 0x00],
    // 91  '{'
    [0x30, 0x40, 0x40, 0xc0, 0x40, 0x40, 0x30, 0x00],
    // 92  '|'
    [0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x00],
    // 93  '}'
    [0x60, 0x10, 0x10, 0x30, 0x10, 0x10, 0x60, 0x00],
    // 94  '~'
    [0x00, 0x00, 0x48, 0xa8, 0x90, 0x00, 0x00, 0x00],
    // 95  TRIANGLE
    [0x20, 0x20, 0x50, 0x50, 0x88, 0x88, 0xf8, 0x00],
    // 96  ARROW_UP
    [0x20, 0x70, 0xa8, 0x20, 0x20, 0x20, 0x20, 0x00],
    // 97  ARROW_DOWN
    [0x20, 0x20, 0x20, 0x20, 0xa8, 0x70, 0x20, 0x00],
    // 98  ARROW_LEFT
    [0x00, 0x20, 0x40, 0xf8, 0x40, 0x20, 0x00, 0x00],
    // 99  ARROW_RIGHT
    [0x00, 0x20, 0x10, 0xf8, 0x10, 0x20, 0x00, 0x00],
    // 100 DISC
    [0x3c, 0x7e, 0xff, 0xff, 0xff, 0xff, 0x7e, 0x3c],
    // 101 CARET
    [0x40, 0x60, 0x70, 0x78, 0x70, 0x60, 0x40, 0x00],
];

// -- the cells past ASCII ---------------------------------------------------

/// `△` — U+25B3, a base-6 digit in the game this table came from.
pub const TRIANGLE: char = '△';
/// `↑` — U+2191, d-pad up.
pub const ARROW_UP: char = '↑';
/// `↓` — U+2193, d-pad down.
pub const ARROW_DOWN: char = '↓';
/// `←` — U+2190, d-pad left.
pub const ARROW_LEFT: char = '←';
/// `→` — U+2192, d-pad right.
pub const ARROW_RIGHT: char = '→';
/// `●` — U+25CF. A filled circle filling the whole cell: a health dot, and the
/// plate behind a face-button glyph. The one cell that is a shape.
pub const DISC: char = '●';
/// `▸` — U+25B8, which menu row has the focus.
pub const CARET: char = '▸';

/// The non-ASCII cells, in table order after [`ASCII_CELLS`].
pub const SYMBOLS: [char; 7] = [
    TRIANGLE,
    ARROW_UP,
    ARROW_DOWN,
    ARROW_LEFT,
    ARROW_RIGHT,
    DISC,
    CARET,
];

/// The cell a character draws from, or [`None`] for anything outside the table.
pub fn cell_of(c: char) -> Option<usize> {
    match c {
        ' '..='~' => Some(c as usize - ASCII_FIRST as usize),
        _ => SYMBOLS
            .iter()
            .position(|s| *s == c)
            .map(|i| ASCII_CELLS + i),
    }
}

/// Every codepoint this table covers, sorted — the map [`BitmapFont`] wants.
pub fn codepoints() -> Vec<u32> {
    let mut all: Vec<u32> = (ASCII_FIRST as u32..=b'~' as u32)
        .chain(SYMBOLS.iter().map(|c| *c as u32))
        .collect();
    all.sort_unstable();
    all
}

/// One cell as 64 bytes of coverage — 0 or 255, row-major.
///
/// What `tools/font-bake` substitutes when a real typeface turns out not to
/// contain a codepoint the game asked for: a hand-authored bitmap beats a
/// missing glyph, and both beat the wrong one.
pub fn cell_coverage(cell: usize) -> [u8; (CELL * CELL) as usize] {
    let mut out = [0u8; (CELL * CELL) as usize];
    let rows = CELLS[cell];
    for (y, bits) in rows.iter().enumerate() {
        for x in 0..CELL as usize {
            if bits & (0x80 >> x) != 0 {
                out[y * CELL as usize + x] = 255;
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// As a `BitmapFont`
// ---------------------------------------------------------------------------

/// The whole table as a one-size [`FontAsset`], baked at [`UNIT`] so `scale`
/// means exactly what it meant when this was the only font there was.
///
/// Every text cell is the fixed 5 × 7 body with a 6-texel advance —
/// monospaced, and deliberately so; [`DISC`] is the exception and carries the
/// whole 8 × 8, which is what lets it be stretched into a health dot.
pub fn asset() -> FontAsset {
    let (w, h) = (atlas_width(), atlas_height());
    let mut coverage = vec![0u8; (w * h) as usize];
    for cell in 0..CELLS.len() {
        let col = (cell as u32 % COLUMNS) * CELL;
        let top = (cell as u32 / COLUMNS) * CELL;
        let bits = cell_coverage(cell);
        for y in 0..CELL {
            let dst = ((top + y) * w + col) as usize;
            let src = (y * CELL) as usize;
            coverage[dst..dst + CELL as usize].copy_from_slice(&bits[src..src + CELL as usize]);
        }
    }

    let codepoints = codepoints();
    let glyphs = codepoints
        .iter()
        .map(|cp| {
            let c = char::from_u32(*cp).expect("the table holds real codepoints");
            let cell = cell_of(c).expect("…and every one of them has a cell");
            let (x, y) = (
                ((cell as u32 % COLUMNS) * CELL) as u16,
                ((cell as u32 / COLUMNS) * CELL) as u16,
            );
            // The shape uses the whole cell; a letter uses its 5 × 7 body, and a
            // space uses none of it.
            let (width, height) = match c {
                ' ' => (0, 0),
                DISC => (CELL as u16, CELL as u16),
                _ => (GLYPH_W as u16, GLYPH_H as u16),
            };
            Glyph {
                x,
                y,
                width,
                height,
                bearing_x: 0,
                // The ink starts at the top of the cell, and the ascent *is* the
                // body height, so `ascent - bearing_y` is zero and a line draws
                // from `y` exactly as this font always has.
                bearing_y: GLYPH_H as i16,
                advance: ADVANCE as f32,
            }
        })
        .collect();

    FontAsset {
        width: w,
        height: h,
        coverage,
        sizes: vec![BitmapFont {
            atlas_width: w,
            atlas_height: h,
            px: UNIT,
            design_scale: 1.0,
            ascent: GLYPH_H as f32,
            descent: (CELL - GLYPH_H) as f32,
            // The body, not the cell: this is the number every layout written
            // against this font already assumes.
            line_height: GLYPH_H as f32,
            missing_advance: ADVANCE as f32,
            glyphs,
            codepoints,
            kerning: Vec::new(),
            ..BitmapFont::default()
        }],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::texture::TextureHandle;
    use crate::ui::UiBatch;

    #[test]
    fn the_grid_holds_every_cell_and_ascii_lands_where_it_claims() {
        assert!(CELLS.len() <= (COLUMNS * atlas_height() / CELL) as usize);
        assert_eq!(CELLS.len(), ASCII_CELLS + SYMBOLS.len());
        assert_eq!(atlas_width(), 128);
        assert_eq!(atlas_height(), 56);

        assert_eq!(cell_of(' '), Some(0));
        assert_eq!(cell_of('A'), Some('A' as usize - 32));
        assert_eq!(cell_of('~'), Some(ASCII_CELLS - 1));
        assert_eq!(cell_of(TRIANGLE), Some(ASCII_CELLS));
        assert_eq!(cell_of(CARET), Some(ASCII_CELLS + 6));
        // Outside the table entirely — drawn as a gap, never as a wrong glyph.
        assert_eq!(cell_of('é'), None);
        assert_eq!(cell_of('—'), None);
    }

    #[test]
    fn every_visible_glyph_has_ink_and_space_has_none() {
        // The failure this exists for is a hole in the table — a character that
        // silently draws nothing, which on a menu label reads as a typo.
        assert_eq!(CELLS[0], [0; 8], "space is not blank");
        for c in '!'..='~' {
            let cell = cell_of(c).expect("printable ASCII has a cell");
            assert!(
                CELLS[cell].iter().any(|row| *row != 0),
                "{c:?} (cell {cell}) is blank"
            );
        }
        for c in SYMBOLS {
            let cell = cell_of(c).unwrap();
            assert!(CELLS[cell].iter().any(|row| *row != 0), "{c:?} is blank");
        }
    }

    #[test]
    fn a_text_glyph_stays_inside_its_five_by_seven_body() {
        // A letter's cell is cropped to 5 × 7, so ink outside that is ink the
        // atlas holds and no quad will ever sample.
        for c in ' '..='~' {
            let rows = CELLS[cell_of(c).unwrap()];
            assert_eq!(rows[7], 0, "{c:?} has ink on row 7");
            for (y, row) in rows.iter().enumerate() {
                assert_eq!(
                    row & 0b0000_0111,
                    0,
                    "{c:?} has ink past column 5 on row {y}"
                );
            }
        }
    }

    #[test]
    fn the_asset_is_the_right_size_and_lays_out_the_way_it_always_did() {
        let asset = asset();
        assert_eq!(asset.coverage.len(), 128 * 56);
        // Round-trips through the checks a game's loader runs.
        let bytes = asset.to_bytes().expect("encode");
        let back = FontAsset::from_bytes(&bytes).expect("decode");
        assert_eq!(back, asset);

        let image = asset.image(TextureHandle(1));
        assert!(image.is_valid());
        // `crate::ui`: coverage in all four channels, or a tinted glyph comes
        // out with a black fringe.
        for texel in image.rgba.chunks_exact(4) {
            assert!(
                texel == [0, 0, 0, 0] || texel == [255, 255, 255, 255],
                "{texel:?} is neither lit nor unlit"
            );
        }

        let font = &asset.fonts(TextureHandle(1))[0];
        assert_eq!(font.design_scale, 1.0);
        // The numbers the port that grew this table was written against.
        assert_eq!(font.width("HI YOU", 2.0), 72.0);
        assert_eq!(font.ink_width("HI YOU", 2.0), 70.0);
        assert_eq!(font.line_height(2.0), 14.0);

        let mut batch = UiBatch::new();
        let end = font.text(&mut batch, 10.0, 20.0, "HI YOU", 2.0, [1.0; 4]);
        assert_eq!(batch.len(), 5, "the space drew something");
        assert_eq!(end, 10.0 + 72.0);
        assert_eq!(batch.quads[0].rect, [10.0, 20.0, 10.0, 14.0]);
    }

    #[test]
    fn a_cells_coverage_matches_its_bits() {
        // 'A' has ink at its apex: row 0, column 2.
        let bits = cell_coverage(cell_of('A').unwrap());
        assert_eq!(bits[2], 255);
        assert_eq!(bits[0], 0);
        // …and the disc fills its whole bottom row's middle.
        let disc = cell_coverage(cell_of(DISC).unwrap());
        assert_eq!(disc[2 * CELL as usize], 255);
    }
}
