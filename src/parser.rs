use nom:: {
    IResult,
    bytes::complete::*,
    character::complete::char,
    combinator::map_res,
    sequence::tuple
};
use crate::bf::Statement;
use nom::multi::many0;
use nom::branch::alt;
use crate::bf::Statement::{Inc, Dec, Next, Prev, Out, In, Loop};
use nom::sequence::terminated;
use std::borrow::Cow;
use std::collections::HashSet;
use once_cell::sync::Lazy;
use nom::combinator::all_consuming;

static BF_CHARS: Lazy<HashSet<char>> = Lazy::new(|| {
    vec!['[', ']', ',', '.', '+', '-', '>', '<']
        .into_iter()
        .collect()
});

pub fn bf_chars(i: &str) -> Cow<str> {
    if i.chars().all(|c| BF_CHARS.contains(&c)) {
        Cow::Borrowed(i)
    } else {
        i.chars()
            .filter(|c| BF_CHARS.contains(c))
            .collect::<String>()
            .into()
    }
}

pub fn program(i: &str) -> IResult<&str, Vec<Statement>> {
    all_consuming(
        stmts
    )(i)
}

pub fn stmts(i: &str) -> IResult<&str, Vec<Statement>> {
    many0(
        alt((
            inc,
            dec,
            next,
            prev,
            out,
            inp,
            loup
            ))
    )(i)
}

pub fn char_op(s: char, stmt: Statement, i: &str) -> IResult<&str, Statement> {
    let (i, _) = char(s)(i)?;
    Ok((i, stmt))
}

pub fn inc(i: &str) -> IResult<&str, Statement> {
    char_op('+', Inc(1), i)
}

pub fn dec(i: &str) -> IResult<&str, Statement> {
    char_op('-', Dec(1), i)
}

pub fn next(i: &str) -> IResult<&str, Statement> {
    char_op('>', Next(1), i)
}

pub fn prev(i: &str) -> IResult<&str, Statement> {
    char_op('<', Prev(1), i)
}

pub fn out(i: &str) -> IResult<&str, Statement> {
    char_op('.', Out, i)
}

pub fn inp(i: &str) -> IResult<&str, Statement> {
    char_op(',', In, i)
}

pub fn loup(i: &str) -> IResult<&str, Statement> {
    let (i, _) = char('[')(i)?;
    let (i, v) = terminated(
        stmts,
        char(']')
    )(i)?;

    Ok((i, Loop(v)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bf::Context;

    #[test]
    fn parse_adder() {
        let (_, s) = program("[[><-]>[->+<]>.<]").unwrap();
        assert_eq!(s.len(), 1);
        let mut ctx = Context::with_state(vec![1, 30, 30]);
        ctx.exec_many(s).unwrap();
        assert_eq!(ctx.data()[2], 60);
        println!("{:?}", &ctx)
    }
}