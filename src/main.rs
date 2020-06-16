use clap::{App, Arg};
use std::io::{stdin, Read};
use bf::parser::{bf_chars, program};
use bf::{exec_many, optimize};
use inkwell::context::Context;
use bf::jit::CodeGen;
use inkwell::passes::{PassManager, PassManagerBuilder};
use inkwell::OptimizationLevel;
use inkwell::targets::{TargetMachine, Target, InitializationConfig, TargetTriple, RelocMode, CodeModel, FileType};


fn main() -> anyhow::Result<()> {
    let m = App::new("bf")
        .arg(Arg::with_name("source-file")
            .value_name("SOURCE")
            .index(1)
            .required(true))
        .arg(Arg::with_name("optimize")
            .short('o'))
        .arg(Arg::with_name("dump")
            .short('d'))
        .arg(Arg::with_name("jit")
            .short('j'))
        .get_matches();

    let f = m.value_of("source-file").unwrap();
    let text = std::fs::read_to_string(f)?;

    let bf_text = bf_chars(&text);
    let sw = stopwatch::Stopwatch::start_new();
    let (_, mut s) = program(&bf_text).expect("Unable to parse");
    if m.is_present("optimize") {
        s = optimize(s);
    }

    let compile;
    if m.is_present("dump") {
        println!("{:?}", &s);
    }
    let exec_start;

    if m.is_present("jit") {
        println!("Jitting...");
        let ctx = Context::create();
        let gen = CodeGen::new(&ctx);
        let func = gen.jit_bf(&s).unwrap();
        let passes = PassManager::create(());

        Target::initialize_all(&InitializationConfig::default());
        let target = Target::from_triple(&TargetMachine::get_default_triple()).unwrap();
        let host = TargetMachine::get_host_cpu_name().to_string();
        let features = TargetMachine::get_host_cpu_features().to_string();
        let tm = target.create_target_machine(
            &TargetMachine::get_default_triple(),
            &host,
            &features,
            OptimizationLevel::Aggressive,
            RelocMode::Static,
            CodeModel::JITDefault
        ).unwrap();

        if m.is_present("optimize") {
            let pm = PassManagerBuilder::create();
            pm.set_optimization_level(OptimizationLevel::Default);
            pm.populate_module_pass_manager(&passes);
            passes.add_promote_memory_to_register_pass();
            passes.add_aggressive_inst_combiner_pass();
            passes.add_loop_vectorize_pass();
        } else {
            let pm = PassManagerBuilder::create();
            pm.set_optimization_level(OptimizationLevel::Less);
            pm.populate_module_pass_manager(&passes);
        }

        let dump = m.is_present("dump");
        if dump {
            gen.module.print_to_file(format!("./{}.unopt.ir", f)).unwrap();
        }
        if passes.run_on(&gen.module) && dump {
            gen.module.print_to_file(format!("./{}.opt.ir", f)).unwrap();
            let asm_dest = format!("./{}.opt.S", f);
            tm.write_to_file(&gen.module, FileType::Assembly, asm_dest.as_ref()).unwrap();
        }
        compile = sw.elapsed_ms();
        let mut ctx = [0i8; 30000];
        let p = ctx.as_mut_ptr();
        println!("EXECUTING JIT!");
        exec_start = sw.elapsed_ms();
        unsafe { func.call(p); }
        println!();
        println!("{:?}", &ctx[..16]);
    } else {
        compile = sw.elapsed_ms();
        exec_start = sw.elapsed_ms();
        exec_many(&s)?;
    }
    let exec = sw.elapsed_ms();
    println!("Compilation took {}ms, execution took {}ms", compile, exec - exec_start);
    Ok(())
}
