use clap::{App, Arg};
use std::io::{stdin, Read};
use bf::parser::{bf_chars, program};
use bf::exec_many;



fn main() -> anyhow::Result<()> {
    let m = App::new("bf")
        .arg(Arg::with_name("source-file")
            .value_name("SOURCE")
            .index(1)
            .required(true))
        .get_matches();

    let f = m.value_of("source-file").unwrap();
    let text = std::fs::read_to_string(f)?;

    let bf_text = bf_chars(&text);
    let (_, s) = program(&bf_text).expect("Unable to parse");
    exec_many(&s)?;
    Ok(())
}
