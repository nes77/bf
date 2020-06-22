use std::io;
use crate::bf::Error::OutOfBounds;
use std::io::Read;

pub mod panicking;

#[derive(Debug, Clone)]
pub enum Statement {
    Next(usize),
    Prev(usize),
    Inc(u64),
    Dec(u64),
    Out,
    In,
    Loop(Vec<Statement>),
    // Indirectly encoded statements
    Clear,
    // [-], [+]
    // Take value at current position, add it to value at other position
    AddOffset { mul: i64, offset: usize },
    SearchZero { stride: isize },
}

impl Statement {
    pub fn is_next(&self) -> bool {
        matches!(self, Statement::Next(_))
    }

    pub fn is_prev(&self) -> bool {
        matches!(self, Statement::Prev(_))
    }

    pub fn is_inc(&self) -> bool {
        matches!(self, Statement::Inc(_))
    }

    pub fn is_dec(&self) -> bool {
        matches!(self, Statement::Dec(_))
    }
}

pub fn optimize(stmts: impl AsRef<[Statement]>) -> Vec<Statement> {
    peephole_optimization(constant_fold(stmts))
}

pub fn constant_fold(stmts: impl AsRef<[Statement]>) -> Vec<Statement> {
    let mut out = Vec::new();
    let stmts = stmts.as_ref();

    let mut idx = 0usize;
    while idx < stmts.len() {
        if let Statement::Loop(l) = &stmts[idx] {
            out.push(
                Statement::Loop(constant_fold(l))
            );
            idx += 1;
            continue;
        } else if matches!(&stmts[idx], Statement::In | Statement::Out | Statement::Clear) {
            out.push(stmts[idx].clone());
            idx += 1;
            continue;
        }

        {
            let next_cnt = stmts[idx..].iter()
                .take_while(|s| s.is_next())
                .count();

            if next_cnt != 0 {
                out.push(Statement::Next(next_cnt));
                idx += next_cnt;
                continue;
            }
        }

        {
            let prev_cnt = stmts[idx..].iter()
                .take_while(|s| s.is_prev())
                .count();

            if prev_cnt != 0 {
                out.push(Statement::Prev(prev_cnt));
                idx += prev_cnt;
                continue;
            }
        }

        {
            let dec_cnt = stmts[idx..].iter()
                .take_while(|s| s.is_dec())
                .count();

            if dec_cnt != 0 {
                out.push(Statement::Dec(dec_cnt as u64));
                idx += dec_cnt;
                continue;
            }
        }

        {
            let inc_cnt = stmts[idx..].iter()
                .take_while(|s| s.is_inc())
                .count();

            if inc_cnt != 0 {
                out.push(Statement::Inc(inc_cnt as u64));
                idx += inc_cnt;
                continue;
            }
        }
    }

    // println!("Reduced by {} statements", stmts.len() - out.len());
    out
}

pub fn peephole_optimization(stmts: impl AsRef<[Statement]>) -> Vec<Statement> {
    stmts.as_ref().iter()
        .flat_map(|s| {
            match s {
                Statement::Loop(l) => {
                    match l.as_slice() {
                        [Statement::Dec(1)] | [Statement::Inc(1)] => vec![Statement::Clear],
                        [Statement::Dec(1), Statement::Next(n), Statement::Inc(inc), Statement::Prev(m)] if n == m => vec![Statement::AddOffset { mul: *inc as i64, offset: *n }, Statement::Clear],
                        [Statement::Prev(n)] => vec![Statement::SearchZero { stride: (*n as isize) * -1 }],
                        [Statement::Next(n)] => vec![Statement::SearchZero { stride: (*n as isize) }],
                        _ => vec![Statement::Loop(peephole_optimization(l))]
                    }
                }
                s => vec![s.clone()]
            }
        }).collect()
}

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("IO error occurred: {0}")]
    IO(#[from] io::Error),
    #[error("Attempted operation out of bounds at idx: {0}")]
    OutOfBounds(usize),
}

pub fn exec(s: Statement) -> Result<(), Error> {
    let mut ctx = Context::new();
    ctx.exec(&s)
}

pub fn exec_many(s: &[Statement]) -> Result<(), Error> {
    let mut ctx = Context::new();
    ctx.exec_many(s)?;
    println!("\n{:?}", ctx);
    Ok(())
}

#[derive(Debug)]
pub struct Context {
    data: Vec<i8>,
    idx: usize,
}

impl Context {
    pub fn new() -> Self {
        Context { data: vec![0], idx: 0 }
    }

    pub fn next(&mut self) {
        self.idx += 1;
        if self.idx >= self.data.len() {
            self.data.push(0);
        }
    }

    pub fn adv(&mut self, a: usize) {
        self.idx += a;
        if self.idx >= self.data.len() {
            self.data.resize(self.idx + 1, 0i8)
        }
    }

    pub fn ret(&mut self, a: usize) {
        self.idx = self.idx.saturating_sub(a);
    }

    pub fn with_state(v: Vec<i8>) -> Self {
        Context {
            data: v,
            idx: 0,
        }
    }

    pub fn prev(&mut self) {
        self.idx = self.idx.saturating_sub(1);
    }

    pub fn data(&self) -> &[i8] {
        &self.data
    }

    pub fn inc(&mut self) -> Result<(), Error> {
        self.data[self.idx] = self.data[self.idx].wrapping_add(1);
        Ok(())
    }

    pub fn inc_many(&mut self, a: u8) -> Result<(), Error> {
        let d = self.data[self.idx] as u8;
        let d = d.wrapping_add(a);

        self.data[self.idx] = d as i8;
        Ok(())
    }

    pub fn dec(&mut self) -> Result<(), Error> {
        self.data[self.idx] = self.data[self.idx].wrapping_sub(1);
        Ok(())
    }

    pub fn dec_many(&mut self, a: u8) -> Result<(), Error> {
        let d = self.data[self.idx] as u8;
        let d = d.wrapping_sub(a);

        self.data[self.idx] = d as i8;
        Ok(())
    }

    pub fn out(&self) -> Result<(), Error> {
        let d = self.data[self.idx];
        print!("{}", d as u8 as char);
        Ok(())
    }

    pub fn inp(&mut self) -> Result<(), Error> {
        let r = &mut self.data[self.idx];
        let mut res = [0u8];
        io::stdin().read_exact(&mut res).map_err(Error::from)?;
        *r = res[0] as i8;
        Ok(())
    }

    pub fn cur(&self) -> Result<i8, Error> {
        Ok(self.data[self.idx])
    }

    pub fn clear(&mut self) {
        self.data[self.idx] = 0;
    }

    pub fn exec(&mut self, s: &Statement) -> Result<(), Error> {
        match s {
            Statement::Next(a) => Ok(self.adv(*a)),
            Statement::Prev(a) => Ok(self.ret(*a)),
            Statement::Inc(a) => self.inc_many(*a as u8),
            Statement::Dec(a) => self.dec_many(*a as u8),
            Statement::Out => self.out(),
            Statement::In => self.inp(),
            Statement::Clear => Ok(self.clear()),
            Statement::Loop(l) => {
                while self.cur()? != 0 {
                    self.exec_many(l)?;
                }

                Ok(())
            }
            Statement::AddOffset { mul, offset } => {
                let mul = *mul;
                let offset = *offset;
                let val = (mul * (self.cur()? as i64));
                self.adv(offset as usize);

                let v = self.cur()? as i64 + val;
                self.data[self.idx] = v as i8;
                self.ret(offset as usize);

                Ok(())
            }
            Statement::SearchZero { stride } => {
                while self.cur()? != 0 {
                    if *stride < 0 {
                        self.ret((*stride * -1) as usize);
                    } else {
                        self.adv(*stride as usize);
                    }
                }

                Ok(())
            }
        }
    }

    pub fn exec_many(&mut self, blk: impl AsRef<[Statement]>) -> Result<(), Error> {
        let prog = blk.as_ref();

        prog.iter().try_for_each(|s| {
            // println!("{:?}: {}", s, self.idx);
            self.exec(s)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bf::Statement::{Dec, Next, Inc, Prev};

    #[test]
    fn adder() {
        let mut ctx = Context {
            idx: 0,
            data: vec![10, 20],
        };

        let prog = Statement::Loop(vec![Dec(1), Next(1), Inc(1), Prev(1)]);

        ctx.exec(&prog).unwrap();

        assert_eq!(ctx.data[1], 30);
    }

    #[test]
    fn optimized() {
        let mut ctx = Context {
            idx: 0,
            data: vec![0, 20],
        };

        let prog = Statement::Loop(vec![Dec(1), Dec(1), Dec(1), Dec(1), Inc(1), Inc(1), Inc(1), Inc(1), Dec(1)]);
        let opt = constant_fold(vec![Inc(1), prog]);


        ctx.exec_many(&opt).unwrap();

        assert_eq!(ctx.data[0], 0);
    }
}