use std::io::{stdin, Read};
use std::process::abort;
use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::builder::Builder;
use inkwell::execution_engine::{ExecutionEngine, JitFunction};
use crate::Statement;
use inkwell::values::{AggregateValue, ArrayValue, PointerValue, FunctionValue};
use inkwell::{AddressSpace, OptimizationLevel, IntPredicate};
use inkwell::passes::PassManagerSubType;

#[no_mangle]
pub extern fn write_char(i: i8) {
    print!("{}", i as u8 as char);
}

#[no_mangle]
pub extern fn read_char() -> i8 {
    let mut buf = [0u8];
    if let Err(e) = stdin().read_exact(&mut buf) {
        eprintln!("{}", e);
        abort();
    };

    buf[0] as i8
}

pub type BFExecFn = unsafe extern "C" fn(*mut i8) -> ();

pub struct CodeGen<'ctx> {
    pub context: &'ctx Context,
    pub module: Module<'ctx>,
    pub builder: Builder<'ctx>,
    pub execution_engine: ExecutionEngine<'ctx>,
}

impl<'ctx> CodeGen<'ctx> {
    pub fn new(ctx: &'ctx Context, opt: OptimizationLevel) -> CodeGen<'ctx> {
        let module = ctx.create_module("bf_exec");
        CodeGen {
            context: ctx,
            builder: ctx.create_builder(),
            execution_engine: module.create_jit_execution_engine(opt).unwrap(),
            module,
        }
    }

    pub fn jit_bf(&self, stmts: impl AsRef<[Statement]>) -> Option<JitFunction<BFExecFn>> {
        let i8_type = self.context.i8_type();
        let void_type = self.context.void_type();
        let index_type = self.context.ptr_sized_int_type(self.execution_engine.get_target_data(), None);
        let fn_type = void_type.fn_type(&[i8_type.ptr_type(AddressSpace::Generic).into()], false);
        let func = self.module.add_function("jit_bf", fn_type, None);

        let writef = self.module.add_function("write_char", void_type.fn_type(&[i8_type.into()], false), None);
        let readf = self.module.add_function("read_char", i8_type.fn_type(&[], false), None);
        let f = format!("llvm.usub.sat.{}", index_type.print_to_string().to_string());
        let _ = self.module.add_function(&f, index_type.fn_type(&[index_type.into(), index_type.into()], false), None);
        self.execution_engine.add_global_mapping(&writef, write_char as usize);
        self.execution_engine.add_global_mapping(&readf, read_char as usize);

        let entry = self.context.append_basic_block(func, "entry");
        self.builder.position_at_end(entry);

        let data = func.get_nth_param(0)?.into_pointer_value();
        let pos = self.builder.build_alloca(index_type, "pos");
        let data_array = self.builder.build_array_alloca(i8_type, index_type.const_int(30000, false), "data");
        let memset_ty = void_type.fn_type(&[data_array.get_type().into(), i8_type.into(), self.context.i32_type().into(), self.context.bool_type().into()], false);
        self.module.add_function("llvm.memset.p0i8.i32", memset_ty, None);
        self.builder.build_call(self.module.get_function("llvm.memset.p0i8.i32").unwrap(), &[data_array.into(), i8_type.const_zero().into(), self.context.i32_type().const_int(30000, false).into(), self.context.bool_type().const_zero().into()], "cleardata");

        self.builder.build_store(pos, index_type.const_zero());

        stmts.as_ref().iter().for_each(|s| self.compile_stmt(func, data_array, pos, s));

        self.builder.build_memcpy(data, 1, data_array, 1, index_type.const_int(30000, false)).unwrap();
        self.builder.build_return(None);

        unsafe { self.execution_engine.get_function("jit_bf").ok() }
    }

    fn compile_stmt(&self, func: FunctionValue, data: PointerValue, pos: PointerValue, s: &Statement) {
        let index_type = self.context.ptr_sized_int_type(self.execution_engine.get_target_data(), None);
        let i8_type = self.context.i8_type();
        let i64_type = self.context.i64_type();

        match s {
            Statement::Next(u) => {
                let cur_val = self.builder.build_load(pos, "cur_pos");
                let new_val = self.builder.build_int_add(cur_val.into_int_value(), index_type.const_int(*u as u64, false), "new_pos");
                self.builder.build_store(pos, new_val);
            }
            Statement::Prev(u) => {
                let cur_val = self.builder.build_load(pos, "cur_pos");
                let f = format!("llvm.usub.sat.{}", index_type.print_to_string().to_string());
                let ssub = self.module.get_function(&f).unwrap();
                let new_val = self.builder.build_call(ssub, &[cur_val, index_type.const_int(*u as u64, false).into()], "new_pos")
                    .try_as_basic_value().left().unwrap().into_int_value();
                self.builder.build_store(pos, new_val);
            }
            Statement::Inc(u) => {
                let cur_pos = self.builder.build_load(pos, "cur_pos");
                let loc = unsafe { self.builder.build_gep(data, &[cur_pos.into_int_value()], "data_pos") };
                let cur_val = self.builder.build_load(loc, "cur_val");
                let new_val = self.builder.build_int_add(cur_val.into_int_value(), i8_type.const_int(*u as u64, false), "new_val");
                self.builder.build_store(loc, new_val);
            }
            Statement::Dec(u) => {
                let cur_pos = self.builder.build_load(pos, "cur_pos");
                let loc = unsafe { self.builder.build_gep(data, &[cur_pos.into_int_value()], "data_pos") };
                let cur_val = self.builder.build_load(loc, "cur_val");
                let new_val = self.builder.build_int_sub(cur_val.into_int_value(), i8_type.const_int(*u as u64, false).into(), "new_val");
                self.builder.build_store(loc, new_val);
            }
            Statement::Out => {
                let f = self.module.get_function("write_char").unwrap();
                let cur_pos = self.builder.build_load(pos, "cur_pos");
                let loc = unsafe { self.builder.build_gep(data, &[cur_pos.into_int_value()], "data_pos") };
                let cur_val = self.builder.build_load(loc, "cur_val").into_int_value();
                self.builder.build_call(f, &[cur_val.into()], "printed");
            }
            Statement::In => {
                let f = self.module.get_function("read_char").unwrap();
                let cur_pos = self.builder.build_load(pos, "cur_pos");
                let loc = unsafe { self.builder.build_gep(data, &[cur_pos.into_int_value()], "data_pos") };
                let new_val = self.builder.build_call(f, &[], "new_val").try_as_basic_value().left().unwrap();
                self.builder.build_store(loc, new_val);
            }
            Statement::Loop(l) => {
                let loop_bb = self.context.append_basic_block(func, "loop");
                self.builder.build_unconditional_branch(loop_bb);
                self.builder.position_at_end(loop_bb);
                let cur_pos = self.builder.build_load(pos, "cur_pos");
                let loc = unsafe { self.builder.build_gep(data, &[cur_pos.into_int_value()], "data_pos") };
                let cur_val = self.builder.build_load(loc, "cur_val");
                let body_loop = self.context.append_basic_block(func, "bodyloop");
                let after_loop = self.context.append_basic_block(func, "afterloop");
                let comp = self.builder.build_int_compare(IntPredicate::NE, cur_val.into_int_value(), i8_type.const_zero(), "comp");
                self.builder.build_conditional_branch(comp, body_loop, after_loop);
                self.builder.position_at_end(body_loop);
                l.iter().for_each(|stmt| self.compile_stmt(func, data, pos, stmt));
                self.builder.build_unconditional_branch(loop_bb);

                self.builder.position_at_end(after_loop);
            }
            Statement::Clear => {
                let cur_pos = self.builder.build_load(pos, "cur_pos");
                let loc = unsafe { self.builder.build_gep(data, &[cur_pos.into_int_value()], "data_pos") };
                self.builder.build_store(loc, i8_type.const_zero());
            }
            Statement::AddOffset { mul, offset } => {
                // Load cur val
                let cur_pos = self.builder.build_load(pos, "cur_pos");
                let loc = unsafe { self.builder.build_gep(data, &[cur_pos.into_int_value()], "data_pos") };
                let cur_val = self.builder.build_load(loc, "cur_val").into_int_value();
                // Multiply by mul
                let add_val = self.builder.build_int_s_extend_or_bit_cast(cur_val, i64_type, "add_val");
                let add_val = self.builder.build_int_mul(add_val, i64_type.const_int(*mul as u64, true), "mul_val");
                let store_loc = unsafe {self.builder.build_gep(loc, &[index_type.const_int(*offset as u64, false)], "store_pos")};
                let cur_val = self.builder.build_load(store_loc, "old_val");
                let cur_val = self.builder.build_int_s_extend_or_bit_cast(cur_val.into_int_value(), i64_type, "old_val");
                let new_val = self.builder.build_int_add(cur_val, add_val, "new_val");
                // Store at offset
                let store_val = self.builder.build_int_truncate_or_bit_cast(new_val, i8_type, "store_size_val");
                self.builder.build_store(store_loc, store_val);
            },
            Statement::SearchZero { stride } => {
                let body = self.context.append_basic_block(func, "search_body");
                let condition = self.context.append_basic_block(func, "cond_block");
                let after_loop = self.context.append_basic_block(func, "after_loop");
                self.builder.build_unconditional_branch(condition);
                self.builder.position_at_end(condition);
                let cur_pos = self.builder.build_load(pos, "cur_pos");
                let loc = unsafe { self.builder.build_gep(data, &[cur_pos.into_int_value()], "data_pos") };
                let cur_val = self.builder.build_load(loc, "cur_val");
                let comp = self.builder.build_int_compare(IntPredicate::NE, cur_val.into_int_value(), i8_type.const_zero(), "comp");
                self.builder.build_conditional_branch(comp, body, after_loop);
                self.builder.position_at_end(body);
                let old_pos = self.builder.build_load(pos, "cur_pos").into_int_value();
                let new_pos = self.builder.build_int_add(old_pos, index_type.const_int(*stride as u64, true), "new_pos");
                self.builder.build_store(pos, new_pos);
                self.builder.build_unconditional_branch(condition);
                self.builder.position_at_end(after_loop);
            }
        }
    }
}