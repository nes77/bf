use std::io;
use crate::bf::Error::OutOfBounds;
use std::io::Read;

#[derive(Debug, Clone)]
pub enum Statement {
    Next,
    Prev,
    Inc,
    Dec,
    Out,
    In,
    Loop(Vec<Statement>)
}

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("IO error occurred: {0}")]
    IO(#[from] io::Error),
    #[error("Attempted operation out of bounds at idx: {0}")]
    OutOfBounds(usize)
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
    idx: usize
}

impl Context {
    pub fn new() -> Self {
        Context {data: vec![0], idx: 0}
    }

    pub fn next(&mut self) {
        self.idx += 1;
        if self.idx >= self.data.len() {
            self.data.push(0);
        }
    }

    pub fn with_state(v: Vec<i8>) -> Self {
        Context {
            data: v,
            idx: 0
        }
    }

    pub fn prev(&mut self) {
        self.idx -= 1;
    }

    pub fn data(&self) -> &[i8] {
        &self.data
    }

    pub fn inc(&mut self) -> Result<(), Error> {
        if self.idx >= self.data.len() {
            return Err(OutOfBounds(self.idx))
        }
        self.data[self.idx] = self.data[self.idx].wrapping_add(1);
        Ok(())
    }

    pub fn dec(&mut self) -> Result<(), Error> {
        if self.idx >= self.data.len() {
            return Err(OutOfBounds(self.idx))
        }
        self.data[self.idx] = self.data[self.idx].wrapping_sub(1);
        Ok(())
    }

    pub fn out(&self) -> Result<(), Error> {
        let d = self.data.get(self.idx);
        d.ok_or(OutOfBounds(self.idx))
            .map(|i| print!("{}", *i as u8 as char))
    }

    pub fn inp(&mut self) -> Result<(), Error> {
        let r = self.data.get_mut(self.idx);
        r.ok_or(OutOfBounds(self.idx))
            .and_then(|d| {
                let mut res = [0u8];
                io::stdin().read_exact(&mut res).map_err(Error::from)?;
                *d = res[0] as i8;
                Ok(())
            })
    }

    pub fn cur(&self) -> Result<i8, Error> {
        self.data.get(self.idx).ok_or(OutOfBounds(self.idx))
            .map(|i| *i)
    }

    pub fn exec(&mut self, s: &Statement) -> Result<(), Error> {
        // println!("{:?}", &s);
        match s {
            Statement::Next => Ok(self.next()),
            Statement::Prev => Ok(self.prev()),
            Statement::Inc => self.inc(),
            Statement::Dec => self.dec(),
            Statement::Out => self.out(),
            Statement::In => self.inp(),
            Statement::Loop(l) => {
                while self.cur()? != 0 {
                    l.iter().try_for_each(|s| self.exec(s))?;
                }

                Ok(())
            },
        }
    }

    pub fn exec_many(&mut self, blk: impl AsRef<[Statement]>) -> Result<(), Error> {
        let prog = blk.as_ref();

        prog.iter().try_for_each(|s| self.exec(s))
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
            data: vec![10, 20]
        };

        let prog = Statement::Loop(vec![Dec, Next, Inc, Prev]);

        ctx.exec(&prog).unwrap();

        assert_eq!(ctx.data[1], 30);

    }

}