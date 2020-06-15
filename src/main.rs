use clap::{App, Arg};
use std::io::{stdin, Read};
use bf::parser::{bf_chars, program};
use bf::{exec_many, optimize};



fn main() -> anyhow::Result<()> {
    let m = App::new("bf")
        .arg(Arg::with_name("source-file")
            .value_name("SOURCE")
            .index(1)
            .required(true))
        .arg(Arg::with_name("optimize")
            .short('o'))
        .get_matches();

    let f = m.value_of("source-file").unwrap();
    let text = std::fs::read_to_string(f)?;

    let bf_text = bf_chars(&text);
    let sw = stopwatch::Stopwatch::start_new();
    let (_, mut s) = program(&bf_text).expect("Unable to parse");
    if m.is_present("optimize") {
        s = optimize(s);
    }
    let compile = sw.elapsed_ms();
    exec_many(&s)?;
    let exec = sw.elapsed_ms();
    println!("Compilation took {}ms, execution took {}ms", compile, exec);
    Ok(())
}
