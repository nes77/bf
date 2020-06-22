use num_traits::{Num, Signed, FromPrimitive, WrappingAdd, ToPrimitive};
use crate::Statement;
use std::io::{stdin, Read};
use nom::lib::std::fmt::{Debug, Formatter};

const NUM_CELLS: usize = (64 * 1024);

macro_rules! impl_static_ctx {
    ($name:ident, $num:ty) => {
        #[derive(Copy, Clone)]
pub struct $name {
    data: [$num; NUM_CELLS],
    pos: usize
}

impl $name {
    pub fn new() -> Self {
        $name {
            data: [0; NUM_CELLS],
            pos: 0
        }
    }

    pub fn with_state(data: impl AsRef<[$num]>) -> Self {
        let mut d = [0 as $num; NUM_CELLS];
        let data = data.as_ref();
        data.iter()
            .take(NUM_CELLS)
            .enumerate()
            .for_each(|(i, datum)| {d[i] = *datum;});

        $name {
            data: d,
            pos: 0
        }
    }

    #[inline]
    pub fn cur(&self) -> $num {
        self.data[self.pos]
    }

    #[inline]
    pub fn adj_pos(&mut self, offset: isize) {
        self.pos = (self.pos as isize + offset) as usize;
    }

    #[inline]
    pub fn adj_val(&mut self, val: i64) {
        self.data[self.pos] = ((self.cur() as i64).wrapping_add(val)) as $num;
    }

    #[inline]
    pub fn clear(&mut self) {
        self.data[self.pos] = 0;
    }

    pub fn data(&self) -> &[$num] {
        &self.data
    }

    pub fn inp(&mut self) {
        let mut dest = [0u8; 1];
        stdin().read_exact(&mut dest).unwrap();
        self.data[self.pos] = dest[0] as $num
    }

    pub fn out(&self) {
        print!("{}", self.cur() as u8 as char)
    }

    pub fn exec(&mut self, s: &Statement) {
        match s {
            Statement::Next(i) => {
                self.adj_pos(*i as isize)
            },
            Statement::Prev(i) => {
                self.adj_pos(-(*i as isize))
            },
            Statement::Inc(i) => {
                self.adj_val(*i as i64)
            },
            Statement::Dec(i) => {
                self.adj_val(-(*i as i64))
            },
            Statement::Out => {
                self.out()
            },
            Statement::In => {
                self.inp()
            },
            Statement::Loop(l) => {
                while self.cur() != 0 {
                    self.exec_many(l)
                }
            },
            Statement::Clear => {
                self.clear()
            },
            Statement::AddOffset { mul, offset } => {
                let c = (self.cur() as i64).wrapping_mul(*mul);
                let cur_dest_val = self.data[self.pos + offset] as i64;
                self.data[self.pos + offset] = c.wrapping_add(cur_dest_val) as $num;
            },
            Statement::SearchZero { stride } => {
                while self.cur() != 0 {
                    self.adj_pos(*stride)
                }
            },
        }
    }

    pub fn exec_many(&mut self, stmts: impl AsRef<[Statement]>) {
        let stmts = stmts.as_ref();
        stmts.iter().for_each(|s| self.exec(s))
    }
}

impl Debug for $name {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}, {}", &self.data[..32], self.pos)
    }
}
    };
}



impl_static_ctx!(StaticContext8, i8);
impl_static_ctx!(StaticContext16, i16);
impl_static_ctx!(StaticContext32, i32);
impl_static_ctx!(StaticContext64, i64);