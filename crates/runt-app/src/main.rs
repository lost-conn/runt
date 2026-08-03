fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    {
        // `runt_core::DEFAULT_LOG_FILTER` rather than a bare "info": wgpu's
        // backend layers narrate their startup at info and drown everything the
        // engine says. `RUST_LOG` still wins.
        env_logger::Builder::from_env(
            env_logger::Env::default().default_filter_or(runt_core::DEFAULT_LOG_FILTER),
        )
        .init();
        runt_app::run();
    }
}
