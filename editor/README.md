# runt-editor

Native [rinch](../../../personal/rinch)-based editor for runt scenes. Own cargo
workspace (rinch's forked wgpu-27/winit-0.31 patches replicated here; they
coexist with runt-core's wgpu 30 — see DESIGN.md §10).

## Run

```
cargo run -p runt-editor            # ~75 s cold build, ~2 s warm
```

## Known issue: blank window / "Paint error: surface lost" on Wayland

rinch's forked **winit 0.31-beta Wayland backend** can fail to present under
KDE/Wayland (endless `surface lost` / `Surface error: Timeout`, window stays
blank — looks like a crash). It affects rinch's own examples identically, on
both the AMD and NVIDIA Vulkan drivers, while stock winit 0.30 apps (the game,
the engine demo) present fine in the same session. It has also been observed
to come and go across sessions.

**Workaround — force XWayland:**

```
WAYLAND_DISPLAY= cargo run -p runt-editor
```

Verified stable via X11. The real fix belongs in rinch's winit fork (its
Wayland patches are the fork's raison d'être — DnD support), not in runt.

## Debug driving

`RINCH_DEBUG_PORT=9931 cargo run -p runt-editor`, then use
`tools/rinchctl.py 9931 screenshot|click|dom|query|…` — note rinch's bundled
`rinch-test` CLI speaks the wrong protocol to this server; rinchctl.py speaks
the real one (length-prefixed frames + handshake).
